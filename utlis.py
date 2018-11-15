import tensorflow as tf
import numpy as np
import math
import os,random
import scipy.io as io
import scipy.ndimage as nd

def bn_my(name, x, in_train):
    shape = [i.value for i in x.get_shape()]
    with tf.variable_scope(name):
        scale = tf.Variable((tf.constant(1.0,tf.float32,shape[-1])))
        add = tf.Variable((tf.constant(0.0,tf.float32,shape[-1])))
        mean,var = tf.nn.moments(x,[0,1,2])
        ema = tf.train.ExponentialMovingAverage(0.99,name='EMA')
        def train():
            opt = ema.apply([mean,var])
            with tf.control_dependencies([opt]):
                return tf.identity(mean),tf.identity(var)
        mean_,var_ = tf.cond(in_train,train,lambda:(ema.average(mean),ema.average(var)))
        norm = tf.nn.batch_normalization(x,mean_,var_,add,scale)
        return norm
        

def BN(name, x, train):
    with tf.variable_scope(name):
        return tf.contrib.layers.batch_norm(x,trainable=train,updates_collections=None)

def relu(name, x):
    with tf.variable_scope(name):
        return tf.nn.relu(x)
    
def lrelu(name, x):
    with tf.variable_scope(name):
        return tf.nn.leaky_relu(x)

def accuracy(r_out,f_out,b):  #  输入是 [b,1,1,1,1]
    error = tf.reduce_sum(tf.cast(tf.less(r_out,0.5),tf.float32))
    correct = tf.reduce_sum(tf.cast(tf.less(f_out,0.5),tf.float32))
    Cor = 0.5 + (correct-error)/(2*b)
    return Cor

def create_file(batch,frames,save_dir,):
    for i in range(batch):
        for j in range(frames):
            dir_ = save_dir+'epoch-'+str(num_ep)+'_sample-'+str(num_b)
            if not os.path.exists(dir_):
                os.makedirs(dir_)
            np.savetxt(dir_+'/'+'B'+str(i)+'_frame'+str(j)+'.txt',arrs[i,j,:,:,0])
    print('save_success')

def getVoxelFromMat(path, cube_len):
    voxels = io.loadmat(path)['instance']
    voxels = np.pad(voxels,(1,1),'constant',constant_values=(0,0))
    if cube_len != 32 and cube_len == 64:
        voxels = nd.zoom(voxels, (2,2,2), mode='constant', order=0)
    return voxels

def get_test_im(path, cube_len):
    volumeBatch = np.asarray([getVoxelFromMat(path, cube_len)],dtype=np.bool)
    volumes = volumeBatch[...,np.newaxis].astype(np.float)  
    return volumes
