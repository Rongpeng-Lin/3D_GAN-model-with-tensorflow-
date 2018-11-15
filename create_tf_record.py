import scipy.io as io
import numpy as np
import scipy.ndimage as nd
import os,random
import tensorflow as tf
import sys
import argparse,sys

def shuffle(names,max_epoch):   # 我自己加的，因为后续程序会使每张图片的出现次数不一样！！！
    Names = names
    retu = []
    for i in range(max_epoch):
        random.shuffle(Names)
        retu += Names
    return retu


def getVoxelFromMat(path, cube_len):
    voxels = io.loadmat(path)['instance']
    voxels = np.pad(voxels,(1,1),'constant',constant_values=(0,0))
    if cube_len != 32 and cube_len == 64:
        voxels = nd.zoom(voxels, (2,2,2), mode='constant', order=0)
    return voxels


# def getAll(obj='chair',train=True, is_local=False, cube_len=64,max_epoch):
def getAll(obj, train, cube_len, max_epoch, LOCAL_PATH):
    objPath = LOCAL_PATH + obj + '/30/'
    objPath += 'train/' if train else 'test/'
    fileList = [f for f in os.listdir(objPath) if f.endswith('.mat')]
    FileList = shuffle(fileList,max_epoch)
    volumeBatch = np.asarray([getVoxelFromMat(objPath + f, cube_len) for f in FileList],dtype=np.bool)
    return volumeBatch


def get_one_class(name, if_train, cube_len, max_epoch, LOCAL_PATH):
    volumes = getAll(name, if_train, cube_len, max_epoch, LOCAL_PATH)
    volumes = volumes[...,np.newaxis].astype(np.float)   # volumes：[64,64,64,1]的矩阵列表
    return volumes
    


def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def image_to_tfexample(image_data):
    return tf.train.Example(features=tf.train.Features(feature={   
      'image': bytes_feature(image_data),
    }))


def _convert_dataset(name, if_train, cube_len, max_epoch, LOCAL_PATH, TFRECORD_DIR, split_name):
    assert split_name in ['train', 'test']
    
    V = get_one_class(name, if_train, cube_len, max_epoch, LOCAL_PATH)
    
    assert np.shape(V[0])[0] == cube_len

    with tf.Session() as sess:
                            
        output_filename = TFRECORD_DIR+'/'+split_name + '.tfrecords'
        
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            for i,Array in enumerate(V):
                sys.stdout.write('\r>> Converting image %d/%d' % (i+1, len(V)))
                sys.stdout.flush()

                array = Array.astype(np.uint8)
                
                image_data = array.tobytes()

                example = image_to_tfexample(image_data)
                
                tfrecord_writer.write(example.SerializeToString())
                
    sys.stdout.write('\n')
    sys.stdout.flush()
    
def main(args):                  
    _convert_dataset(args.name, args.if_train, args.cube_len, args.max_epoch, args.LOCAL_PATH, args.TFRECORD_DIR, args.split_name)   
    return True         

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help='class name', default="chair")
    parser.add_argument('--LOCAL_PATH', type=str, help='mat file dir', default="D:/3DShapeNets/volumetric_data/")
    parser.add_argument('--TFRECORD_DIR', type=str, help='tf_record save dir', default="E:/zdelete")
    parser.add_argument('--split_name', type=str, help='train or test', default="train")
    parser.add_argument('--cube_len', type=int, help='the lenth of a cube', default=64)
    parser.add_argument('--max_epoch', type=int, help='the max epoches', default=5)
    parser.add_argument('--if_train', type=bool, help='the lenth of a cube', default=True)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
