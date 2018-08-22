import tensorflow as tf
import os
from PIL import Image
import models.base_model as model

def create_image_tfrecord(tfrecord_filepath, data_basepath,class_dict, bestnum=1000):
    num = 0
    recordfilenum = 0
    # tfrecords格式文件名
    tfrecord_filename = ("traindata.tfrecords-%.3d" % recordfilenum)
    writer = tf.python_io.TFRecordWriter(os.path.join(tfrecord_filepath),tfrecord_filename)
    for class_name in os.listdir(data_basepath):
        if class_name not in class_dict.keys(): continue
        for image_name in os.listdir(os.path.join(data_basepath,class_name)):
            num = num + 1
            if num > bestnum:
                num = 1
                recordfilenum = recordfilenum + 1
                tfrecord_filename = ("traindata.tfrecords-%.3d" % recordfilenum)
                writer = tf.python_io.TFRecordWriter(tfrecord_filepath + tfrecord_filename)
            img = Image.open(os.path.join(data_basepath,class_name,image_name))
            size = img.size
            img_raw = img.tobytes()  # 将图片转化为二进制格式
            example = tf.train.Example(
                features=tf.train.Features(feature={
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[class_dict[class_name]])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                    'img_width': tf.train.Feature(int64_list=tf.train.Int64List(value=[size[0]])),
                    'img_height': tf.train.Feature(int64_list=tf.train.Int64List(value=[size[1]]))
                }))
            writer.write(example.SerializeToString())  # 序列化为字符串
    writer.close()


def read_image_tfrecord(tfrecord_path, image_type,label_type):
    features = model.read_tfrecord(tfrecord_path,{
                                       'label': tf.FixedLenFeature([], tf.int64),
                                       'img_raw' : tf.FixedLenFeature([], tf.string),
                                       'img_width': tf.FixedLenFeature([], tf.int64),
                                       'img_height': tf.FixedLenFeature([], tf.int64),
                                   })
    image = tf.decode_raw(features['img_raw'], image_type)
    height = tf.cast(features['img_height'], tf.int32)
    width = tf.cast(features['img_width'], tf.int32)
    label = tf.cast(features['label'], tf.int32)
    return image,height,width,label


