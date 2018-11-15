import tensorflow as tf
import numpy as np
import math
import os,random


def conv3d(name, x, depth, kers, outs, s, padd):
    shape = [i.value for i in x.get_shape()]
    ker = int(math.sqrt(kers))
    pad = "SAME" if padd else "VALID"
    with tf.variable_scope(name):
        w = tf.get_variable('w',
                            [depth,ker,ker,shape[-1],outs],
                            tf.float32,
                            tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('b',
                            [outs],
                            tf.float32,
                            tf.initializers.constant(0.))
        conv = tf.nn.conv3d(x,w,[1,s,s,s,1],pad) + b
        return conv
    
    #     weights['wg1'] = tf.get_variable("wg1", shape=[4, 4, 4, 512, 200], initializer=xavier_init)
    #     g_1 = tf.nn.conv3d_transpose(z, weights['wg1'], (batch_size,4,4,4,512), strides=[1,1,1,1,1], padding="VALID")
                          
def conv3d_trans(name, x, deepth, kers, outs, s, padd, out_shape):
    shape = [i.value for i in x.get_shape()]
    ker = int(math.sqrt(kers))
    pad = "SAME" if padd else "VALID"
    with tf.variable_scope(name):
        w = tf.get_variable('w',
                            [deepth,ker,ker,outs,shape[-1]],
                            tf.float32,
                            tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('b',
                            [outs],
                            tf.float32,
                            tf.initializers.constant(0.))
        conv_trans = tf.nn.conv3d_transpose(x, w, out_shape, [1,s,s,s,1], pad) + b
        return conv_trans
    
def Save(arrs,save_dir,num_ep,num_b):
    batch,frames = np.shape(arrs)[0:2]
    for i in range(batch):
        for j in range(frames):
            dir_ = save_dir+'output_ims'+'/'+'epoch-'+str(num_ep)+'_sample-'+str(num_b)
            np.savetxt(dir_+'/'+'B'+str(i)+'_frame'+str(j)+'.txt',arrs[i,j,:,:,0])
    print('save_success')
    
def Save_test(arrs,save_dir,num_ep,num_b):
    batch,frames = np.shape(arrs)[0:2]
    for i in range(batch):
        for j in range(frames):
            dir_ = save_dir+'output_ims'+'/'+'epoch-'+str(num_ep)+'_sample-'+str(num_b)
            if not os.path.exists(dir_):
                os.makedirs(dir_)
            np.savetxt(dir_+'/'+'B'+str(i)+'_frame'+str(j)+'.txt',arrs[i,j,:,:,0])
    print('save_success')
    
def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   
    features = tf.parse_single_example(serialized_example,features={'image' : tf.FixedLenFeature([],tf.string)})
    image = tf.decode_raw(features['image'], tf.uint8)
    image = tf.cast(tf.reshape(image,[64,64,64,1]),tf.float32)
    return image

def read_from_txt(file_dir):
    zeros = np.zeros([len(os.listdir(file_dir)),64,64,1])
    for i,name in enumerate(os.listdir(file_dir)):
        txt_dir = file_dir+name
        a = np.loadtxt(txt_dir)
        zeros[i,:,:,0] = a
    return zeros
