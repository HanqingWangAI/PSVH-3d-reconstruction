import sys
import numpy as np
import tensorflow as tf
import os
import voxel
from PIL import Image
import time
from Refinement_utils import *



voxel_size = 32
img_h = 128
img_w = 128
vector_channel = 1024

threshold = 0.4

cates = ["04256520", "02691156", "03636649", "04401088",
            "04530566", "03691459", "03001627", "02933112",
            "04379243", "03211117", "02958343", "02828884", "04090263"] # The object categories

dic = {"04256520": "sofa", "02691156": "airplane", "03636649": "lamp", "04401088": "telephone",
            "04530566": "vessel", "03691459": "loudspeaker", "03001627": "chair", "02933112": "cabinet",
            "04379243": "table", "03211117": "display", "02958343": "car", "02828884": "bench", "04090263": "rifle"}


def encoder_residual_block(input, layer_id, num_layers=2, channels=None):
    input_shape = input.get_shape()
    last_channel = int(input_shape[-1])
    last_layer = input
    batch_size = int(input_shape[0])

    wd_res = tf.get_variable("wres%d" % layer_id, shape=[1, 1, last_channel, channels],
                             initializer=tf.contrib.layers.xavier_initializer())
    wb_res = tf.get_variable("bres%d" % layer_id, shape=[channels], initializer=tf.zeros_initializer())
    res = tf.nn.conv2d(input, wd_res, strides=[1, 1, 1, 1], padding='SAME')
    res = tf.nn.bias_add(res, wb_res)
    res = lrelu(res)

    for i in range(num_layers):
        wd_conv = tf.get_variable("wd%d_%d" % (layer_id, i), shape=[3, 3, last_channel, channels],
                                  initializer=tf.contrib.layers.xavier_initializer())
        wb_conv = tf.get_variable("wb%d_%d" % (layer_id, i), shape=[channels], initializer=tf.zeros_initializer())
        last_layer = tf.nn.conv2d(last_layer, wd_conv, strides=[1, 1, 1, 1], padding='SAME')
        last_layer = tf.nn.bias_add(last_layer, wb_conv)
        last_layer = lrelu(last_layer)
        last_channel = channels

    output = res + last_layer
    return output

def encoder(input, reuse=False):
    # print(input.get_shape()[0])
    batch_size = int(input.get_shape()[0])
    input = tf.reshape(input, shape=[batch_size, img_h, img_w, 3])
    layer_id = 1
    shortcuts = []
    with tf.variable_scope("encoder", reuse=reuse):
        wd00 = tf.get_variable("wd00", shape=[7, 7, 3, 96], initializer=tf.contrib.layers.xavier_initializer())
        bd00 = tf.get_variable("bd00", shape=[96], initializer=tf.zeros_initializer())
        conv0a = tf.nn.conv2d(input, wd00, strides=[1, 1, 1, 1], padding='SAME')
        conv0a = tf.nn.bias_add(conv0a, bd00)

        wd01 = tf.get_variable("wd01", shape=[3, 3, 96, 96], initializer=tf.contrib.layers.xavier_initializer())
        bd01 = tf.get_variable("bd01", shape=[96], initializer=tf.zeros_initializer())
        conv0b = tf.nn.conv2d(conv0a, wd01, strides=[1, 1, 1, 1], padding='SAME')
        conv0b = tf.nn.bias_add(conv0b, bd01)

        pool1 = tf.nn.max_pool(conv0b, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        shortcuts.append(pool1)

        conv1 = encoder_residual_block(pool1, layer_id, 2, 128)
        pool2 = tf.nn.max_pool(conv1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        shortcuts.append(pool2)
        layer_id += 1

        conv2 = encoder_residual_block(pool2, layer_id, 2, 256)
        pool3 = tf.nn.max_pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        shortcuts.append(pool3)
        layer_id += 1

        wd30 = tf.get_variable("wd30", shape=[3, 3, 256, 256], initializer=tf.contrib.layers.xavier_initializer())
        bd30 = tf.get_variable("bd30", shape=[256], initializer=tf.zeros_initializer())
        conv3a = tf.nn.conv2d(pool3, wd30, strides=[1, 1, 1, 1], padding='SAME')
        conv3a = tf.nn.bias_add(conv3a, bd30)

        wd31 = tf.get_variable("wd31", shape=[3, 3, 256, 256], initializer=tf.contrib.layers.xavier_initializer())
        bd31 = tf.get_variable("bd31", shape=[256], initializer=tf.zeros_initializer())
        conv3b = tf.nn.conv2d(conv3a, wd31, strides=[1, 1, 1, 1], padding='SAME')
        conv3b = tf.nn.bias_add(conv3b, bd31)

        pool4 = tf.nn.max_pool(conv3b, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        shortcuts.append(pool4)
        layer_id += 1

        conv4 = encoder_residual_block(pool4, layer_id, 2, 256)
        pool5 = tf.nn.max_pool(conv4, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        shortcuts.append(pool5)
        layer_id += 1

        conv5 = encoder_residual_block(pool5, layer_id, 2, 256)
        pool6 = tf.nn.max_pool(conv5, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        feature_map = pool6

        pool6 = tf.reduce_mean(pool6, [1, 2])
        wfc = tf.get_variable("wfc", shape=[256, 1024], initializer=tf.contrib.layers.xavier_initializer())
        feature = tf.matmul(pool6, wfc)

        w_e = tf.get_variable("w_euler", shape=[1024, 3], initializer=tf.contrib.layers.xavier_initializer())
        euler_angle = tf.matmul(feature, w_e)

        w_st = tf.get_variable('w_ft', shape=[1024, 3], initializer=tf.contrib.layers.xavier_initializer())
        st = tf.matmul(feature, w_st)

        print('pool1', pool1)
        print('pool2', pool2)
        print('pool3', pool3)
        print('pool4', pool4)
        print('pool5', pool5)
        print('pool6', pool6)
        print('feature', feature)
        print('feature_map', feature_map)

        return feature, feature_map, euler_angle, st, shortcuts

def encoder_angle(input, reuse=False):
    batch_size = int(input.get_shape()[0])
    input = tf.reshape(input, shape=[batch_size, img_h, img_w, 3])
    layer_id = 1
    shortcuts = []
    eulers_cates = {}
    st_cates = {}
    with tf.variable_scope("encoder", reuse=reuse):
        wd00 = tf.get_variable("wd00", shape=[7, 7, 3, 96], initializer=tf.contrib.layers.xavier_initializer())
        bd00 = tf.get_variable("bd00", shape=[96], initializer=tf.zeros_initializer())
        conv0a = tf.nn.conv2d(input, wd00, strides=[1, 1, 1, 1], padding='SAME')
        conv0a = tf.nn.bias_add(conv0a, bd00)

        wd01 = tf.get_variable("wd01", shape=[3, 3, 96, 96], initializer=tf.contrib.layers.xavier_initializer())
        bd01 = tf.get_variable("bd01", shape=[96], initializer=tf.zeros_initializer())
        conv0b = tf.nn.conv2d(conv0a, wd01, strides=[1, 1, 1, 1], padding='SAME')
        conv0b = tf.nn.bias_add(conv0b, bd01)

        pool1 = tf.nn.max_pool(conv0b, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        shortcuts.append(pool1)

        conv1 = encoder_residual_block(pool1, layer_id, 2, 128)
        pool2 = tf.nn.max_pool(conv1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        shortcuts.append(pool2)
        layer_id += 1

        conv2 = encoder_residual_block(pool2, layer_id, 2, 256)
        pool3 = tf.nn.max_pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        shortcuts.append(pool3)
        layer_id += 1

        wd30 = tf.get_variable("wd30", shape=[3, 3, 256, 256], initializer=tf.contrib.layers.xavier_initializer())
        bd30 = tf.get_variable("bd30", shape=[256], initializer=tf.zeros_initializer())
        conv3a = tf.nn.conv2d(pool3, wd30, strides=[1, 1, 1, 1], padding='SAME')
        conv3a = tf.nn.bias_add(conv3a, bd30)

        wd31 = tf.get_variable("wd31", shape=[3, 3, 256, 256], initializer=tf.contrib.layers.xavier_initializer())
        bd31 = tf.get_variable("bd31", shape=[256], initializer=tf.zeros_initializer())
        conv3b = tf.nn.conv2d(conv3a, wd31, strides=[1, 1, 1, 1], padding='SAME')
        conv3b = tf.nn.bias_add(conv3b, bd31)

        pool4 = tf.nn.max_pool(conv3b, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        shortcuts.append(pool4)
        layer_id += 1

        conv4 = encoder_residual_block(pool4, layer_id, 2, 256)
        pool5 = tf.nn.max_pool(conv4, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        shortcuts.append(pool5)
        layer_id += 1

        conv5 = encoder_residual_block(pool5, layer_id, 2, 256)
        pool6 = tf.nn.max_pool(conv5, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        feature_map = pool6

        pool6 = tf.reduce_mean(pool6, [1, 2])
        wfc = tf.get_variable("wfc", shape=[256, 1024], initializer=tf.contrib.layers.xavier_initializer())
        feature = tf.matmul(pool6, wfc)

        print('pool1', pool1)
        print('pool2', pool2)
        print('pool3', pool3)
        print('pool4', pool4)
        print('pool5', pool5)
        print('pool6', pool6)
        print('feature', feature)
        print('feature_map', feature_map)

        return feature, feature_map, shortcuts

def generator(input, shortcuts, reuse=False):
    batch_size = int(input.shape[0])
    strides = [[1, 2, 2, 1],  # 4
               [1, 2, 2, 1],  # 8
               [1, 2, 2, 1],  # 16
               [1, 2, 2, 1],  # 32
               [1, 2, 2, 1],  # 64
               [1, 2, 2, 1]]  # 127

    print(input)

    with tf.variable_scope("ge", reuse=reuse):
        wg1 = tf.get_variable('wg1', shape=[3, 3, 256, 256], initializer=tf.contrib.layers.xavier_initializer())
        bg1 = tf.get_variable('bg1', shape=[256], initializer=tf.zeros_initializer())
        g_1 = tf.nn.conv2d_transpose(input, wg1, [batch_size, 4, 4, 256], strides=strides[0], padding='SAME')
        g_1 = tf.nn.bias_add(g_1, bg1)
        g_1 = lrelu(g_1)
        g_1 = tf.add(g_1, shortcuts[4])

        wg2 = tf.get_variable('wg2', shape=[3, 3, 256, 256], initializer=tf.contrib.layers.xavier_initializer())
        bg2 = tf.get_variable('bg2', shape=[256], initializer=tf.zeros_initializer())
        g_2 = tf.nn.conv2d_transpose(g_1, wg2, [batch_size, 8, 8, 256], strides=strides[1], padding='SAME')
        g_2 = tf.nn.bias_add(g_2, bg2)
        g_2 = lrelu(g_2)
        g_2 = tf.add(g_2, shortcuts[3])

        wg3 = tf.get_variable('wg3', shape=[3, 3, 256, 256], initializer=tf.contrib.layers.xavier_initializer())
        bg3 = tf.get_variable('bg3', shape=[256], initializer=tf.zeros_initializer())
        g_3 = tf.nn.conv2d_transpose(g_2, wg3, [batch_size, 16, 16, 256], strides=strides[2], padding='SAME')
        g_3 = tf.nn.bias_add(g_3, bg3)
        g_3 = lrelu(g_3)
        g_3 = tf.add(g_3, shortcuts[2])

        wg4 = tf.get_variable('wg4', shape=[3, 3, 128, 256], initializer=tf.contrib.layers.xavier_initializer())
        bg4 = tf.get_variable('bg4', shape=[128], initializer=tf.zeros_initializer())
        g_4 = tf.nn.conv2d_transpose(g_3, wg4, [batch_size, 32, 32, 128], strides=strides[3], padding='SAME')
        g_4 = tf.nn.bias_add(g_4, bg4)
        g_4 = lrelu(g_4)
        g_4 = tf.add(g_4, shortcuts[1])

        wg5 = tf.get_variable('wg5', shape=[4, 4, 96, 128], initializer=tf.contrib.layers.xavier_initializer())
        bg5 = tf.get_variable('bg5', shape=[96], initializer=tf.zeros_initializer())
        g_5 = tf.nn.conv2d_transpose(g_4, wg5, [batch_size, 64, 64, 96], strides=strides[4], padding='SAME')
        g_5 = tf.nn.bias_add(g_5, bg5)
        g_5 = lrelu(g_5)
        g_5 = tf.add(g_5, shortcuts[0])

        wg6 = tf.get_variable('wg6', shape=[4, 4, 2, 96], initializer=tf.contrib.layers.xavier_initializer())
        g_6 = tf.nn.conv2d_transpose(g_5, wg6, [batch_size, img_h, img_w, 2], strides=strides[5], padding='SAME')
        mask_softmax = tf.nn.softmax(g_6)
    return g_6, mask_softmax

def decoder(input, reuse=False):
    batch_size = int(input.get_shape()[0])
    strides = [1, 2, 2, 2, 1]
    layer_id = 2
    print(input)
    with tf.variable_scope("decoder", reuse=reuse):
        input = tf.reshape(input, (batch_size, 1, 1, 1, 1024))
        print(input)
        wd = tf.get_variable("wd1", shape=[4, 4, 4, 256, 1024],
                             initializer=tf.contrib.layers.xavier_initializer())
        bd = tf.get_variable("bd1", shape=[256], initializer=tf.zeros_initializer())

        d_1 = tf.nn.conv3d_transpose(input, wd, (batch_size, 4, 4, 4, 256), strides=[1, 1, 1, 1, 1], padding='VALID')
        d_1 = tf.nn.bias_add(d_1, bd)
        d_1 = tf.nn.relu(d_1)

        d_2 = residual_block(d_1, layer_id)
        layer_id += 1

        d_3 = residual_block(d_2, layer_id)
        layer_id += 1

        d_4 = residual_block(d_3, layer_id)
        layer_id += 1

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
        res_softmax = tf.nn.softmax(res)
        print('d6', res)
        return res, res_softmax

def syn_model():
    cates = ['03001627'] # The chair category

    y_vectors = tf.placeholder(shape=[1, img_w, img_h, 3], dtype=tf.float32, name='all_Images')

    with tf.variable_scope('voxel'):
        feature, feature_map, euler, st, shortcuts = encoder(y_vectors, reuse=False)
        voxels, voxels_softmax_before = decoder(feature, reuse=False)

    with tf.variable_scope("mask"):
        feature, feature_map, _, _, shortcuts = encoder(y_vectors, reuse=False)
        mask, mask_softmax = generator(feature_map, shortcuts, reuse=False)

    feature, feature_map, shortcuts = encoder_angle(y_vectors, reuse=False)

    voxel_after_dic = {}

    for i, cate in enumerate(cates):
        if i == 0:
            reuse = False
        else:
            reuse = True

        with tf.variable_scope("angles_trans", reuse=False):
            w_e_1 = tf.get_variable("w_euler_0_%s" % cate, shape=[1024, 512],
                                    initializer=tf.contrib.layers.xavier_initializer())
            e_1 = lrelu(tf.matmul(feature, w_e_1))
            w_e_2 = tf.get_variable("w_euler_1_%s" % cate, shape=[512, 3],
                                    initializer=tf.contrib.layers.xavier_initializer())
            euler = tf.matmul(e_1, w_e_2)

            w_st = tf.get_variable('w_ft_%s' % cate, shape=[1024, 3],
                                   initializer=tf.contrib.layers.xavier_initializer())
            st = tf.matmul(feature, w_st)

        rotation_matrices = get_rotation_matrix_r2n2(euler)
        mask_indexs = scale_trans_r2n2(st)
        projection = cast(mask_softmax[..., 0], mask_indexs, rotation_matrices=rotation_matrices)
        c1 = voxels_softmax_before[..., 0]
        c2 = projection
        c3 = c1 - c1 * c2
        c4 = c2 - c1 * c2

        feedin = tf.stack([c1, c2, c3, c4], axis=4)

        feature_vector, shortcuts = refine_encoder(feedin, reuse=reuse)
        voxels, voxels_softmax_after = refine_decoder(feature_vector, shortcuts, reuse=reuse)

        voxel_after_dic[cate] = voxels_softmax_after[..., 0]


    return voxels_softmax_before[...,0], voxel_after_dic['03001627'], y_vectors

def real_model():
    cates = ['03001627'] # The chair category

    # weight_path = 'voxel/100001.cptk'
    # refine_path = os.path.join('checkpoint_cross_categories_4_Adam_refine','70501.cptk')
    


    y_vectors = tf.placeholder(shape=[1, img_w, img_h, 3], dtype=tf.float32, name='all_Images')
    # m_vectors = tf.placeholder(shape=[batchsize, img_w, img_h, 2], dtype=tf.float32, name='all_Masks')
    # e_vectors = tf.placeholder(shape=[batchsize, 3], dtype=tf.float32, name='all_Angles')
    # st_vectors = tf.placeholder(shape=[batchsize, 3], dtype=tf.float32, name='all_scale_translation')

    with tf.variable_scope('voxel'):
        feature, feature_map, euler, st, shortcuts = encoder(y_vectors, reuse=False)
        voxels, voxels_softmax_before = decoder(feature, reuse=False)

    with tf.variable_scope("mask"):
        feature, feature_map, _, _, shortcuts = encoder(y_vectors, reuse=False)
        mask, mask_softmax = generator(feature_map, shortcuts, reuse=False)

    feature, feature_map, shortcuts = encoder_angle(y_vectors, reuse=False)



    voxel_after_dic = {}

    for i,cate in enumerate(cates):
        if i == 0:
            reuse = False
        else:
            reuse = True

        with tf.variable_scope("angles_trans", reuse=False):
            w_e_1 = tf.get_variable("w_euler_0_%s" % cate, shape=[1024, 512],
                                    initializer=tf.contrib.layers.xavier_initializer())
            e_1 = lrelu(tf.matmul(feature, w_e_1))
            w_e_2 = tf.get_variable("w_euler_1_%s" % cate, shape=[512, 3],
                                    initializer=tf.contrib.layers.xavier_initializer())
            euler = tf.matmul(e_1, w_e_2)

            w_st = tf.get_variable('w_ft_%s' % cate, shape=[1024, 3],
                                   initializer=tf.contrib.layers.xavier_initializer())
            st = tf.matmul(feature, w_st)
            st = tf.stack([st[..., 0] * 10, st[..., 1], st[..., 2]], axis=-1)

        rotation_matrices = get_rotation_matrix_voc(euler)
        mask_indexs = scale_trans_voc(st)
        masks = rotate_mask_voc(mask_softmax[..., 0], euler)
        projection = cast(masks, mask_indexs, rotation_matrices=rotation_matrices)
        c1 = voxels_softmax_before[..., 0]
        c2 = projection
        c3 = c1 - c1 * c2
        c4 = c2 - c1 * c2

        feedin = tf.stack([c1, c2, c3, c4], axis=4)

        feature_vector, shortcuts = refine_encoder(feedin, reuse=reuse)
        voxels, voxels_softmax_after = refine_decoder(feature_vector, shortcuts, reuse=reuse)

        voxel_after_dic[cate] = voxels_softmax_after[..., 0]


    return voxels_softmax_before[...,0], voxel_after_dic['03001627'], y_vectors


'''
    choose the model
    'syn' is the model trained on synthetic data
    'real' is the model finetuned on the PASCAL VOC 3D+ (real images) dataset
'''
def examples(model='syn'):
    from voxel import voxel2obj
    cates = ['chair']
    
    if not model in ['syn','real']:
        print("The model '%s' is not in 'syn' or 'real'"%model)
        return
    
    weight_path = os.path.join('models','%s_model'%model,'model.cptk')
    source_path = '%s_data'%model
    dest_path = 'res_%s_data'%model

    if model == 'syn':
        before, after, img_input = syn_model()
    else:
        before, after, img_input = real_model()

    params = tf.trainable_variables()
    saver = tf.train.Saver(var_list=params)

    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    for cate in cates:
        files = [file for file in os.listdir(os.path.join(source_path,cate)) if file[-3:]=='png']

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess,weight_path)
        for file in files:
            filepath = os.path.join(source_path,'chair',file)
            img = Image.open(filepath).resize((img_h,img_w))
            img = np.array(img).astype(np.float32)/255.
            img = img.reshape([1, img_h, img_w, 3])

            v_before, v_after = sess.run([before,after],feed_dict={img_input:img})
            v_before = v_before.squeeze() > threshold
            v_after = v_after.squeeze() > threshold
          
            voxel2obj('%s/%s_before.obj'%(dest_path,file[:-4]),v_before)
            voxel2obj('%s/%s_after.obj'%(dest_path,file[:-4]),v_after)

if __name__ == '__main__':
    examples('real')
