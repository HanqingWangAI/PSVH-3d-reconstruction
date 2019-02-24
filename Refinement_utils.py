import numpy as np
import tensorflow as tf
import os
import voxel
from binvox_rw import Voxels
import time
from math import cos
from math import sin
from math import pi
from math import sqrt



voxel_size = 32
img_h = 128
img_w = 128
vector_channel = 1024

id_x = None
id_y = None

def pre_process():
    global id_x, id_y
    dim = voxel_size
    points = []
    dis = 1.2
    focus = img_w * sqrt(3) / 2

    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                points.append([1.0*i/dim - 0.5, 1.0*j/dim-0.5, 1.0*k/dim-0.5])
    points = np.asarray(points)
    points[..., 2] += dis
    for i in range(2):
        points[..., i] *= focus
        points[..., i] /= points[..., 2]

    id_x = np.round(np.clip(points[...,0]+img_w/2,0,img_w-1)).astype(np.int32)
    id_y = np.round(np.clip((points[...,1]+img_w/2),0,img_w-1)).astype(np.int32)
    id_y = img_w - 1 - id_y

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak*x)

def residual_block(input,layer_id,num_layers=2,div=2,unpool=True):
    input_shape = input.get_shape()
    last_channel = input_shape[-1]
    current_channel = int(int(last_channel)/div)


    # reg = tf.contrib.layers.l2_regularizer(scale=0.1)
    # upsampling

    strides = [1,1,1,1,1]
    output_shape = [int(input_shape[0]), int(input_shape[1]), int(input_shape[2]), int(input_shape[3]),
                        current_channel]
    if unpool:
        strides = [1,2,2,2,1]
        output_shape = [int(input_shape[0]), int(input_shape[1]) * 2, int(input_shape[2]) * 2, int(input_shape[3]) * 2,
                        current_channel]

    wd_0 = tf.get_variable("wd%d_0"%layer_id,shape=[3,3,3,current_channel,last_channel],initializer=tf.contrib.layers.xavier_initializer())
    bd_0 = tf.get_variable("bd%d_0"%layer_id,shape=[current_channel],initializer=tf.zeros_initializer())
    d_0 = tf.nn.conv3d_transpose(value=input, filter=wd_0, output_shape=output_shape, strides=strides, padding='SAME')
    d_0 = tf.nn.bias_add(d_0, bd_0)
    d_0 = tf.nn.relu(d_0)

    last_layer = d_0


    for i in range(1,num_layers+1):
        wd = tf.get_variable("wd%d_%d" % (layer_id,i), shape=[3, 3, 3, current_channel, current_channel],
                               initializer=tf.contrib.layers.xavier_initializer())
        bd = tf.get_variable("bd%d_%d" % (layer_id,i), shape=[current_channel],initializer=tf.zeros_initializer())
        d = tf.nn.conv3d(last_layer, filter=wd, strides=[1, 1, 1, 1, 1], padding='SAME')
        d = tf.nn.bias_add(d, bd)
        d = tf.nn.relu(d)
        last_layer = d

    return tf.add(last_layer, d_0)

def refine_encoder(input, reuse=False):
    strides = [1, 2, 2, 2, 1]
    shortcuts = []
    shortcuts.append(tf.stack([input[...,0],1-input[...,0]],axis=4))

    with tf.variable_scope("refine_encoder",reuse=reuse):
        we1 = tf.get_variable("we1",shape=[5,5,5,4,32],initializer=tf.contrib.layers.xavier_initializer())
        be1 = tf.get_variable("be1",shape=[32],initializer=tf.zeros_initializer())

        e_1 = tf.nn.conv3d(input, we1, strides=[1,1,1,1,1], padding="SAME")
        e_1 = tf.nn.bias_add(e_1, be1)
        #d_1 = tf.contrib.layers.batch_norm(d_1, is_training=phase_train)
        e_1 = lrelu(e_1)
        shortcuts.append(e_1) # 32 32 32 32

        we2 = tf.get_variable("we2", shape=[3, 3, 3, 32, 64], initializer=tf.contrib.layers.xavier_initializer())
        be2 = tf.get_variable("be2", shape=[64], initializer=tf.zeros_initializer())

        e_2 = tf.nn.conv3d(e_1, we2, strides=strides, padding="SAME")
        e_2 = tf.nn.bias_add(e_2, be2)
        #d_2 = tf.contrib.layers.batch_norm(d_2, is_training=phase_train)
        e_2 = lrelu(e_2)
        shortcuts.append(e_2) # 16 16 16 64


        we3 = tf.get_variable("we3", shape=[3, 3, 3, 64, 128], initializer=tf.contrib.layers.xavier_initializer())
        be3 = tf.get_variable("be3", shape=[128], initializer=tf.zeros_initializer())
        e_3 = tf.nn.conv3d(e_2, we3, strides=strides, padding="SAME")
        e_3 = tf.nn.bias_add(e_3, be3)
        # d_2 = tf.contrib.layers.batch_norm(d_2, is_training=phase_train)
        e_3 = lrelu(e_3)
        shortcuts.append(e_3)  # 8 8 8 128

        we4 = tf.get_variable("we4", shape=[3, 3, 3, 128, 256], initializer=tf.contrib.layers.xavier_initializer())
        be4 = tf.get_variable("be4", shape=[256], initializer=tf.zeros_initializer())
        e_4 = tf.nn.conv3d(e_3, we4, strides=strides, padding="SAME")
        e_4 = tf.nn.bias_add(e_4, be4)
        # d_2 = tf.contrib.layers.batch_norm(d_2, is_training=phase_train)
        e_4 = lrelu(e_4)
        shortcuts.append(e_4)  # 4 4 4 256

        we5 = tf.get_variable("we5", shape=[4, 4, 4, 256, 512], initializer=tf.contrib.layers.xavier_initializer())
        be5 = tf.get_variable("be5", shape=[512], initializer=tf.zeros_initializer())
        e_5 = tf.nn.conv3d(e_4, we5, strides=[1,1,1,1,1], padding="VALID")
        e_5 = tf.nn.bias_add(e_5, be5)
        # d_2 = tf.contrib.layers.batch_norm(d_2, is_training=phase_train)
        e_5 = lrelu(e_5) # 1 1 1 1024

        for _ in shortcuts:
            print(_)
        return e_5, shortcuts

def refine_encoder_novisualhull(input, reuse=False):
    strides = [1, 2, 2, 2, 1]
    shortcuts = []
    shortcuts.append(tf.stack([input[...,0],1-input[...,0]],axis=4))

    with tf.variable_scope("refine_encoder",reuse=reuse):
        we1 = tf.get_variable("we1",shape=[5,5,5,2,32],initializer=tf.contrib.layers.xavier_initializer())
        be1 = tf.get_variable("be1",shape=[32],initializer=tf.zeros_initializer())

        e_1 = tf.nn.conv3d(input, we1, strides=[1,1,1,1,1], padding="SAME")
        e_1 = tf.nn.bias_add(e_1, be1)
        #d_1 = tf.contrib.layers.batch_norm(d_1, is_training=phase_train)
        e_1 = lrelu(e_1)
        shortcuts.append(e_1) # 32 32 32 32

        we2 = tf.get_variable("we2", shape=[3, 3, 3, 32, 64], initializer=tf.contrib.layers.xavier_initializer())
        be2 = tf.get_variable("be2", shape=[64], initializer=tf.zeros_initializer())

        e_2 = tf.nn.conv3d(e_1, we2, strides=strides, padding="SAME")
        e_2 = tf.nn.bias_add(e_2, be2)
        #d_2 = tf.contrib.layers.batch_norm(d_2, is_training=phase_train)
        e_2 = lrelu(e_2)
        shortcuts.append(e_2) # 16 16 16 64


        we3 = tf.get_variable("we3", shape=[3, 3, 3, 64, 128], initializer=tf.contrib.layers.xavier_initializer())
        be3 = tf.get_variable("be3", shape=[128], initializer=tf.zeros_initializer())
        e_3 = tf.nn.conv3d(e_2, we3, strides=strides, padding="SAME")
        e_3 = tf.nn.bias_add(e_3, be3)
        # d_2 = tf.contrib.layers.batch_norm(d_2, is_training=phase_train)
        e_3 = lrelu(e_3)
        shortcuts.append(e_3)  # 8 8 8 128

        we4 = tf.get_variable("we4", shape=[3, 3, 3, 128, 256], initializer=tf.contrib.layers.xavier_initializer())
        be4 = tf.get_variable("be4", shape=[256], initializer=tf.zeros_initializer())
        e_4 = tf.nn.conv3d(e_3, we4, strides=strides, padding="SAME")
        e_4 = tf.nn.bias_add(e_4, be4)
        # d_2 = tf.contrib.layers.batch_norm(d_2, is_training=phase_train)
        e_4 = lrelu(e_4)
        shortcuts.append(e_4)  # 4 4 4 256

        we5 = tf.get_variable("we5", shape=[4, 4, 4, 256, 512], initializer=tf.contrib.layers.xavier_initializer())
        be5 = tf.get_variable("be5", shape=[512], initializer=tf.zeros_initializer())
        e_5 = tf.nn.conv3d(e_4, we5, strides=[1,1,1,1,1], padding="VALID")
        e_5 = tf.nn.bias_add(e_5, be5)
        # d_2 = tf.contrib.layers.batch_norm(d_2, is_training=phase_train)
        e_5 = lrelu(e_5) # 1 1 1 1024

        for _ in shortcuts:
            print(_)
        return e_5, shortcuts

def refine_decoder(input, shortcuts, reuse = False):
    strides = [1, 2, 2, 2, 1]
    layer_id = 2
    print(input)
    batch_size = int(input.get_shape()[0])
    with tf.variable_scope("refine_decoder", reuse=reuse):
        input = tf.reshape(input, (batch_size, 1, 1, 1, 512))
        wd = tf.get_variable("wd1", shape=[4, 4, 4, 256, 512],
                             initializer=tf.contrib.layers.xavier_initializer())
        bd = tf.get_variable("bd1", shape=[256], initializer=tf.zeros_initializer())

        d_1 = tf.nn.conv3d_transpose(input, wd, (batch_size, 4, 4, 4, 256), strides=[1, 1, 1, 1, 1], padding='VALID')
        d_1 = tf.nn.bias_add(d_1, bd)
        d_1 = tf.nn.relu(d_1)
        d_1 = tf.add(d_1,shortcuts[4])

        d_2 = residual_block(d_1, layer_id) # 8 8 8 128
        layer_id += 1
        d_2 = tf.add(d_2,shortcuts[3])

        d_3 = residual_block(d_2, layer_id) # 16 16 16 64
        layer_id += 1
        d_3 = tf.add(d_3,shortcuts[2])

        d_4 = residual_block(d_3, layer_id) # 32 32 32 32
        layer_id += 1
        d_4 = tf.add(d_4,shortcuts[1])

        d_5 = residual_block(d_4, layer_id, 3, unpool=False)
        layer_id += 1

        last_channel = int(d_5.shape[-1])

        print('d1', d_1)
        print('d2', d_2)
        print('d3', d_3)
        print('d4', d_4)
        print('d5', d_5)

        wd = tf.get_variable("wd6", shape=[3, 3, 3, 2, last_channel],
                             initializer=tf.contrib.layers.xavier_initializer())

        res = tf.nn.conv3d_transpose(d_5, wd, (batch_size, 32, 32, 32, 2), strides=[1, 1, 1, 1, 1], padding='SAME')
        d_6 = res
        res = tf.add(res, shortcuts[0])
        res_softmax = tf.nn.softmax(res)
        print('d6', res)
        return res, res_softmax#,d_5,d_6

def scale_trans(st):
    points = []
    #dis = 1.5
    focus = 138 * sqrt(3) / 2
    dim = voxel_size

    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                points.append([1.0*i/dim - 0.5, 1.0*j/dim-0.5, 1.0*k/dim-0.5])


    points = np.asarray(points)
    #points[..., 2] += dis
    # points[..., 0] += dis

    #for i in range(2):
    #    points[..., i] *= focus
    #    points[..., i] /= points[..., 2]

    batch = int(st.shape[0])

    # points = Con(points)
    # mask_indexs = []
    # for i in range(batch):
    #     scale, transx, transy = st[i,0], st[i,1], st[i,2]
    #     tx = tf.round(tf.clip_by_value((points[...,2])*scale-transx+img_w/2,0,img_w-1))
    #     ty = img_w - 1 - tf.round(tf.clip_by_value((points[...,1])*scale+transy+img_w/2,0,img_w-1))
    #     t_id = ty * img_w + tx + i * img_w * img_h
    #     mask_indexs.append(t_id)
    #
    # ret = tf.stack(mask_indexs)
    # total = 1
    #
    # for _ in ret.shape:
    #     total *= int(_)
    #
    # ret = tf.cast(tf.reshape(ret,[total]),dtype=tf.int32)
    points = Con(points)
    mask_indexs = []
    for i in range(batch):
        dis, transx, transy = st[i,0], st[i,1], st[i,2]
        pointsz = points[..., 2] + dis

        pointsx = (points[..., 0] * focus) / pointsz
        pointsy = (points[..., 1] * focus) / pointsz

        tx = tf.round(tf.clip_by_value((pointsx)-transx+img_w/2,0,img_w-1))
        ty = tf.round(tf.clip_by_value((pointsy)-transy+img_w/2,0,img_w-1))
        t_id = ty * img_w + tx + i * img_w * img_h
        mask_indexs.append(t_id)

    ret = tf.stack(mask_indexs)
    total = 1

    for _ in ret.shape:
        total *= int(_)

    ret = tf.cast(tf.reshape(ret,[total]),dtype=tf.int32)
    return ret

def scale_trans_r2n2(st):
    points = []
    #dis = 1.5
    # focus = 138 * sqrt(3) / 2
    scale = 1
    focus = 157.2275*scale
    dim = voxel_size

    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                points.append([1.0*i/dim - 0.5, 1.0*j/dim-0.5, 1.0*k/dim-0.5])


    points = np.asarray(points)
    #points[..., 2] += dis
    # points[..., 0] += dis

    #for i in range(2):
    #    points[..., i] *= focus
    #    points[..., i] /= points[..., 2]

    batch = int(st.shape[0])

    # points = Con(points)
    # mask_indexs = []
    # for i in range(batch):
    #     scale, transx, transy = st[i,0], st[i,1], st[i,2]
    #     tx = tf.round(tf.clip_by_value((points[...,2])*scale-transx+img_w/2,0,img_w-1))
    #     ty = img_w - 1 - tf.round(tf.clip_by_value((points[...,1])*scale+transy+img_w/2,0,img_w-1))
    #     t_id = ty * img_w + tx + i * img_w * img_h
    #     mask_indexs.append(t_id)
    #
    # ret = tf.stack(mask_indexs)
    # total = 1
    #
    # for _ in ret.shape:
    #     total *= int(_)
    #
    # ret = tf.cast(tf.reshape(ret,[total]),dtype=tf.int32)
    points = Con(points)
    mask_indexs = []
    for i in range(batch):
        dis, transx, transy = st[i,0], st[i,1], st[i,2]
        pointsz = points[..., 2] + dis*scale

        pointsx = (points[..., 0] * focus) / pointsz
        pointsy = (points[..., 1] * focus) / pointsz

        tx = tf.round(tf.clip_by_value((pointsx)-transx+img_w/2,0,img_w-1))
        ty = tf.round(tf.clip_by_value((pointsy)-transy+img_w/2,0,img_w-1))
        t_id = ty * img_w + tx + i * img_w * img_h
        mask_indexs.append(t_id)

    ret = tf.stack(mask_indexs)
    total = 1

    for _ in ret.shape:
        total *= int(_)

    ret = tf.cast(tf.reshape(ret,[total]),dtype=tf.int32)
    return ret

def scale_trans_voc(st):
    points = []
    #dis = 1.5
    scale = 0.5
    focus = 138 * sqrt(3) * 10 * scale
    dim = voxel_size

    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                points.append([1.0*i/dim - 0.5, 1.0*j/dim-0.5, 1.0*k/dim-0.5])


    points = np.asarray(points)
    #points[..., 2] += dis
    # points[..., 0] += dis

    #for i in range(2):
    #    points[..., i] *= focus
    #    points[..., i] /= points[..., 2]

    batch = int(st.shape[0])

    points = Con(points)
    mask_indexs = []
    for i in range(batch):
        dis, transx, transy = st[i,0], st[i,1], st[i,2]
        pointsz = points[..., 2] + dis*scale

        pointsx = (points[..., 0] * focus) / pointsz
        pointsy = (points[..., 1] * focus) / pointsz

        tx = tf.round(tf.clip_by_value((pointsx)-transx+img_w/2,0,img_w-1))
        ty = tf.round(tf.clip_by_value((pointsy)-transy+img_w/2,0,img_w-1))
        t_id = ty * img_w + tx + i * img_w * img_h
        mask_indexs.append(t_id)

    ret = tf.stack(mask_indexs)
    total = 1

    for _ in ret.shape:
        total *= int(_)

    ret = tf.cast(tf.reshape(ret,[total]),dtype=tf.int32)
    return ret

def rotate_mask_voc(mask, angles):
    batch = int(angles.shape[0])
    points = []
    half = img_h/2.
    for i in range(img_h):
        for j in range(img_w):
            points.append([1.0*i-half,1.0*j-half])
    points = Con(np.asarray(points).transpose())

    masks = []
    for i in range(batch):
        theta = 2*pi*angles[i, 2]
        # theta = Con(pi)
        R = tf.reshape(tf.stack([tf.cos(theta), -tf.sin(theta), tf.sin(theta), tf.cos(theta)]), [2, 2])

        rotation = tf.cast(tf.clip_by_value(tf.round(tf.matmul(R, points) + half), 0, img_h - 1),
                           dtype=tf.int32)
        # x.append(tf.reshape(rotation[0, ...], shape=[dim, dim, dim]))
        # y.append(tf.reshape(rotation[1, ...], shape=[dim, dim, dim]))
        # z.append(tf.reshape(rotation[2, ...], shape=[dim, dim, dim]))
        x = rotation[0, ...]
        y = rotation[1, ...]
        index = x*img_h + y
        data = tf.reshape(mask[i,...],[img_h*img_w])
        new_mask = tf.gather(data, index)
        masks.append(tf.reshape(new_mask,[img_w,img_h]))

    masks = tf.stack(masks)
    return masks


def cast(masks, mask_indexs, rotation_matrices):
    dim = voxel_size
    batch_size = int(masks.shape[0])
    mask_shape = masks.shape
    x,y,z = rotation(rotation_matrices)
    total = 1
    for _ in mask_shape:
        total *= int(_)

    masks = tf.reshape(masks, shape=[total])
    data = tf.gather(masks, mask_indexs)

    project_indexs = x*dim*dim+y*dim+z
    project_indexs = tf.reshape(project_indexs, [int(project_indexs.shape[0])*int(project_indexs.shape[1])])

    datas = tf.gather(data, project_indexs)
    datas = tf.reshape(datas, [batch_size,dim,dim,dim])
    return datas

def Con(val):
    return tf.constant(val, dtype=np.float32)

def flip():

    from binvox_rw import read_as_3d_array
    from binvox_rw import Voxels
    with open('model.binvox','rb') as fp:
        vox = read_as_3d_array(fp)

        data = vox.data
        data = np.transpose(data,[2,1,0])
    new_vox = Voxels(data,vox.dims,vox.translate)
    with open('model_transpose.binvox','wb') as fp:
        new_vox.write(fp)

def get_rotation_matrix(angles):
    angles = 2*pi*angles
    batch = int(angles.shape[0])
    matrices = []
    # for i in range(batch):
    #     #pitch, yaw, roll = angles[i, 1]+0.5*pi, angles[i, 2], -angles[i, 0]
    #     pitch, roll, yaw = -angles[i, 1]+pi, angles[i, 2], angles[i, 0]
    #     Rx = tf.reshape(tf.stack([Con(1),Con(0),Con(0), Con(0), tf.cos(roll), -tf.sin(roll), Con(0), tf.sin(roll), tf.cos(roll)]),[3,3])
    #     Ry = tf.reshape(tf.stack([tf.cos(pitch), Con(0), tf.sin(pitch), Con(0), Con(1), Con(0), -tf.sin(pitch), Con(0), tf.cos(pitch)]),[3,3])
    #     Rz = tf.reshape(tf.stack([tf.cos(yaw), -tf.sin(yaw), Con(0), tf.sin(yaw), tf.cos(yaw), Con(0), Con(0), Con(0), Con(1)]), [3,3])
    #     R = tf.matmul(Rz,tf.matmul(Ry,Rx))
    #     matrices.append(R)
    for i in range(batch):
        #pitch, yaw, roll = angles[i, 1]+0.5*pi, angles[i, 2], -angles[i, 0]
        pitch, yaw, roll = angles[i, 1], angles[i, 2], angles[i, 0]
        Rx = tf.reshape(tf.stack([Con(1),Con(0),Con(0), Con(0), tf.cos(roll), -tf.sin(roll), Con(0), tf.sin(roll), tf.cos(roll)]),[3,3])
        Ry = tf.reshape(tf.stack([tf.cos(pitch), Con(0), tf.sin(pitch), Con(0), Con(1), Con(0), -tf.sin(pitch), Con(0), tf.cos(pitch)]),[3,3])
        Rz = tf.reshape(tf.stack([tf.cos(yaw), -tf.sin(yaw), Con(0), tf.sin(yaw), tf.cos(yaw), Con(0), Con(0), Con(0), Con(1)]), [3,3])
        R = tf.matmul(Rx, tf.matmul(Ry,Rz))
        matrices.append(R)

    matrices = tf.stack(matrices)
    return matrices

def get_rotation_matrix_r2n2(angles):
    angles = 2*pi*angles
    batch = int(angles.shape[0])
    matrices = []
    # for i in range(batch):
    #     #pitch, yaw, roll = angles[i, 1]+0.5*pi, angles[i, 2], -angles[i, 0]
    #     pitch, roll, yaw = -angles[i, 1]+pi, angles[i, 2], angles[i, 0]
    #     Rx = tf.reshape(tf.stack([Con(1),Con(0),Con(0), Con(0), tf.cos(roll), -tf.sin(roll), Con(0), tf.sin(roll), tf.cos(roll)]),[3,3])
    #     Ry = tf.reshape(tf.stack([tf.cos(pitch), Con(0), tf.sin(pitch), Con(0), Con(1), Con(0), -tf.sin(pitch), Con(0), tf.cos(pitch)]),[3,3])
    #     Rz = tf.reshape(tf.stack([tf.cos(yaw), -tf.sin(yaw), Con(0), tf.sin(yaw), tf.cos(yaw), Con(0), Con(0), Con(0), Con(1)]), [3,3])
    #     R = tf.matmul(Rz,tf.matmul(Ry,Rx))
    #     matrices.append(R)
    for i in range(batch):
        #pitch, yaw, roll = angles[i, 1]+0.5*pi, angles[i, 2], -angles[i, 0]
        pitch, yaw, roll = -angles[i, 1], angles[i, 2], angles[i, 0]
        Rx = tf.reshape(tf.stack([Con(1),Con(0),Con(0), Con(0), tf.cos(roll), -tf.sin(roll), Con(0), tf.sin(roll), tf.cos(roll)]),[3,3])
        Ry = tf.reshape(tf.stack([tf.cos(pitch), Con(0), tf.sin(pitch), Con(0), Con(1), Con(0), -tf.sin(pitch), Con(0), tf.cos(pitch)]),[3,3])
        Rz = tf.reshape(tf.stack([tf.cos(yaw), -tf.sin(yaw), Con(0), tf.sin(yaw), tf.cos(yaw), Con(0), Con(0), Con(0), Con(1)]), [3,3])
        R = tf.matmul(Rx, tf.matmul(Ry,Rz))
        matrices.append(R)

    matrices = tf.stack(matrices)
    return matrices

def get_rotation_matrix_voc(angles):
    angles = 2*pi*angles
    batch = int(angles.shape[0])
    matrices = []
    # for i in range(batch):
    #     #pitch, yaw, roll = angles[i, 1]+0.5*pi, angles[i, 2], -angles[i, 0]
    #     pitch, roll, yaw = -angles[i, 1]+pi, angles[i, 2], angles[i, 0]
    #     Rx = tf.reshape(tf.stack([Con(1),Con(0),Con(0), Con(0), tf.cos(roll), -tf.sin(roll), Con(0), tf.sin(roll), tf.cos(roll)]),[3,3])
    #     Ry = tf.reshape(tf.stack([tf.cos(pitch), Con(0), tf.sin(pitch), Con(0), Con(1), Con(0), -tf.sin(pitch), Con(0), tf.cos(pitch)]),[3,3])
    #     Rz = tf.reshape(tf.stack([tf.cos(yaw), -tf.sin(yaw), Con(0), tf.sin(yaw), tf.cos(yaw), Con(0), Con(0), Con(0), Con(1)]), [3,3])
    #     R = tf.matmul(Rz,tf.matmul(Ry,Rx))
    #     matrices.append(R)
    for i in range(batch):
        #pitch, yaw, roll = angles[i, 1]+0.5*pi, angles[i, 2], -angles[i, 0]
        pitch, yaw, roll = angles[i, 1], Con(0), angles[i, 0]
        Rx = tf.reshape(tf.stack([Con(1),Con(0),Con(0), Con(0), tf.cos(roll), -tf.sin(roll), Con(0), tf.sin(roll), tf.cos(roll)]),[3,3])
        Ry = tf.reshape(tf.stack([tf.cos(pitch), Con(0), tf.sin(pitch), Con(0), Con(1), Con(0), -tf.sin(pitch), Con(0), tf.cos(pitch)]),[3,3])
        Rz = tf.reshape(tf.stack([tf.cos(yaw), -tf.sin(yaw), Con(0), tf.sin(yaw), tf.cos(yaw), Con(0), Con(0), Con(0), Con(1)]), [3,3])
        R = tf.matmul(Rx, tf.matmul(Ry,Rz))
        matrices.append(R)

    matrices = tf.stack(matrices)
    return matrices

def rotate_and_translate(R, st):
    dim = voxel_size
    batch = int(R.shape[0])
    cords = []
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                cords.append([1.0 * i / dim - 0.5, 1.0 * j / dim - 0.5, 1.0 * k / dim - 0.5])

    cords = np.asarray(cords).transpose()

    points = Con(cords)

    cords = []

    for i in range(batch):
        dis, transx, transy = st[i, 0], st[i, 1], st[i, 2]
        rotation = tf.matmul(R[i], points)
        x = rotation[0, ...] - transx/dis
        y = rotation[1, ...] - transy/dis
        z = rotation[2, ...] + dis
        rotation = tf.transpose(tf.stack([x,y,z],axis=0))
        cords.append(rotation)

    cords = tf.stack(cords,axis=0) # [batchsize, 32*32*32, 3]
    cords = tf.reshape(cords,[batch,dim,dim,dim,3])
    return cords

def DVX(visual_hull):
    visual_hull = tf.expand_dims(visual_hull,axis=-1)
    kernal = np.array([-1,1])
    kx = Con(kernal.reshape([2,1,1,1,1]))
    ky = Con(kernal.reshape([1,2,1,1,1]))
    kz = Con(kernal.reshape([1,1,2,1,1]))

    dx = tf.squeeze(tf.nn.conv3d(visual_hull,kx,[1,1,1,1,1],padding='SAME'))
    dy = tf.squeeze(tf.nn.conv3d(visual_hull,ky,[1,1,1,1,1],padding='SAME'))
    dz = tf.squeeze(tf.nn.conv3d(visual_hull,kz,[1,1,1,1,1],padding='SAME'))

    res = tf.stack([dx,dy,dz],axis=-1)

    return res

def rotation(R):
    dim = voxel_size
    batch = int(R.shape[0])
    cords = []
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                cords.append([1.0 * i / dim - 0.5, 1.0 * j / dim - 0.5, 1.0 * k / dim - 0.5])

    cords = np.asarray(cords).transpose()

    points = Con(cords)

    x = []
    y = []
    z = []

    for i in range(batch):
        rotation = tf.cast(tf.clip_by_value(tf.round((tf.matmul(R[i], points) + 0.5) * dim), 0, dim-1), dtype=tf.int32)
        # x.append(tf.reshape(rotation[0, ...], shape=[dim, dim, dim]))
        # y.append(tf.reshape(rotation[1, ...], shape=[dim, dim, dim]))
        # z.append(tf.reshape(rotation[2, ...], shape=[dim, dim, dim]))
        x.append(rotation[0, ...])
        y.append(rotation[1, ...])
        z.append(rotation[2, ...] + i*dim*dim*dim)

    x = tf.stack(x, axis=0)
    y = tf.stack(y, axis=0)
    z = tf.stack(z, axis=0)

    return x, y, z