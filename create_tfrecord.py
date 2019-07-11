# -*-coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import random
from PIL import Image

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
# 生成字符串型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
# 生成实数型的属性
def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def get_example_nums(tf_records_filenames):
    nums= 0
    for record in tf.python_io.tf_record_iterator(tf_records_filenames):
        nums += 1
    return nums

def load_labels_file(filename,labels_num=1,shuffle=False):
    images=[]
    labels=[]
    with open(filename) as f:
        lines_list=f.readlines()
        if shuffle:
            random.shuffle(lines_list)

        for lines in lines_list:
            line=lines.rstrip().split(' ')
            label=[]
            for i in range(labels_num):
                label.append(int(line[i+1]))
            images.append(line[0])
            labels.append(label)
    return images,labels

def read_image(filename, resize_height, resize_width,normalization=False):
 
    bgr_image = cv2.imread(filename)
    if len(bgr_image.shape)==2:#若是灰度图则转为三通道
        print("Warning:gray image",filename)
        bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2BGR)

    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)#将BGR转为RGB
   
    if resize_height>0 and resize_width>0:
        rgb_image=cv2.resize(rgb_image,(resize_width,resize_height))
    rgb_image=np.asanyarray(rgb_image)
    if normalization:
       
        rgb_image=rgb_image/255.0
  
    return rgb_image


def get_batch_images(images,labels,batch_size,labels_nums,one_hot=False,shuffle=False,num_threads=1):

    min_after_dequeue = 20
    capacity = min_after_dequeue + 3 * batch_size  # 保证capacity必须大于min_after_dequeue参数值
    if shuffle:
        images_batch, labels_batch = tf.train.shuffle_batch([images,labels],
                                                                    batch_size=batch_size,
                                                                    capacity=capacity,
                                                                    min_after_dequeue=min_after_dequeue,
                                                                    num_threads=num_threads)
    else:
        images_batch, labels_batch = tf.train.batch([images,labels],
                                                        batch_size=batch_size,
                                                        capacity=capacity,
                                                        num_threads=num_threads)
    if one_hot:
        labels_batch = tf.one_hot(labels_batch, labels_nums, 1, 0)
    return images_batch,labels_batch

def read_records(filename,resize_height, resize_width,type=None):

    # 创建文件队列,不限读取的数量
    filename_queue = tf.train.string_input_producer([filename])
    # create a reader from file queue
    reader = tf.TFRecordReader()
    # reader从文件队列中读入一个序列化的样本
    _, serialized_example = reader.read(filename_queue)
    # get feature from serialized example
    # 解析符号化的样本
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        }
    )
    tf_image = tf.decode_raw(features['image_raw'], tf.uint8)#获得图像原始的数据
    tf_label = tf.cast(features['label'], tf.int32)
    # PS:恢复原始图像数据,reshape的大小必须与保存之前的图像shape一致,否则出错
    tf_image=tf.reshape(tf_image, [resize_height, resize_width, 3]) # 设置图像的维度
    # 存储的图像类型为uint8,tensorflow训练时数据必须是tf.float32
    if type is None:
        tf_image = tf.cast(tf_image, tf.float32)
    elif type=='normalization':# [1]若需要归一化请使用:
        # 仅当输入数据是uint8,才会归一化[0,255]
        # tf_image = tf.image.convert_image_dtype(tf_image, tf.float32)
        tf_image = tf.cast(tf_image, tf.float32) * (1. / 255.0)  # 归一化
    elif type=='centralization':
        # 若需要归一化,且中心化,假设均值为0.5,请使用:
        tf_image = tf.cast(tf_image, tf.float32) * (1. / 255) - 0.5 #中心化

    return tf_image,tf_label


def create_records(image_dir,file, output_record_dir, resize_height, resize_width,shuffle,log=5):
    '''
    实现将图像原始数据,label,长,宽等信息保存为record文件
    注意:读取的图像数据默认是uint8,再转为tf的字符串型BytesList保存,解析请需要根据需要转换类型
    :param image_dir:原始图像的目录
    :param file:输入保存图片信息的txt文件(image_dir+file构成图片的路径)
    :param output_record_dir:保存record文件的路径
    :param resize_height:
    :param resize_width:
    PS:当resize_height或者resize_width=0是,不执行resize
    :param shuffle:是否打乱顺序
    :param log:log信息打印间隔
    '''
    # 加载文件,仅获取一个label
    images_list, labels_list=load_labels_file(file,1,shuffle)

    writer = tf.python_io.TFRecordWriter(output_record_dir)
    for i, [image_name, labels] in enumerate(zip(images_list, labels_list)):
        image_path=os.path.join(image_dir,images_list[i])
        if not os.path.exists(image_path):
            print('Err:no image',image_path)
            continue
        image = read_image(image_path, resize_height, resize_width)
        image_raw = image.tostring()
        if i%log==0 or i==len(images_list)-1:
            print('------------processing:%d-th------------' % (i))
            print('current image_path=%s' % (image_path),'shape:{}'.format(image.shape),'labels:{}'.format(labels))
        # 这里仅保存一个label,多label适当增加"'label': _int64_feature(label)"项
        label=labels[0]
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': _bytes_feature(image_raw),
            'label': _int64_feature(label)
        }))
        writer.write(example.SerializeToString())
    writer.close()

if __name__ == '__main__':
    # 参数设置

    resize_height = 224  # 指定存储图片高度
    resize_width = 224  # 指定存储图片宽度
    shuffle=True
    log=5
    # 产生train.record文件
    image_dir='dataset/train'
    train_labels = 'dataset/train.txt'  # 图片路径
    train_record_output = 'dataset/record/train224.tfrecords'
    create_records(image_dir,train_labels, train_record_output, resize_height, resize_width,shuffle,log)
    train_nums=get_example_nums(train_record_output)
    print("save train example nums={}".format(train_nums))

    # 产生val.record文件
    image_dir='dataset/val'
    val_labels = 'dataset/val.txt'  # 图片路径
    val_record_output = 'dataset/record/val224.tfrecords'
    create_records(image_dir, val_labels, val_record_output, resize_height, resize_width, shuffle, log)
    val_nums=get_example_nums(val_record_output)
    print("save val example nums={}".format(val_nums))