import tensorflow as tf
import os
from PIL import Image
import models.base_model as model
from tqdm import tqdm
from tensorflow.python.platform import gfile
import numpy as np

BOTTLENECK_TENSOR_SIZE = 2048

BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'

JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
MODEL_DIR = '/media/ai/data/workrooms/pre_model/inception_v3'

MODEL_FILE = 'tensorflow_inception_graph.pb'

def create_image_tfrecord(tfrecord_path,type_name,data_basepath,class_dict, bestnum=1000):
    num = 0
    recordfilenum = 0
    # tfrecords格式文件名
    tfrecord_filename = "%s.tfrecords-%.5d" % (type_name, recordfilenum)
    writer = tf.python_io.TFRecordWriter(os.path.join(tfrecord_path,tfrecord_filename))
    for class_name in os.listdir(data_basepath):
        if class_name not in class_dict.keys(): continue
        for image_name in tqdm(os.listdir(os.path.join(data_basepath,class_name))):
            num = num + 1
            if num > bestnum:
                num = 1
                recordfilenum = recordfilenum + 1
                tfrecord_filename = "%s.tfrecords-%.5d" % (type_name, recordfilenum)
                writer = tf.python_io.TFRecordWriter(os.path.join(tfrecord_path, tfrecord_filename))
            img = Image.open(os.path.join(data_basepath,class_name,image_name))
            size = img.size
            img_raw = img.tobytes()  # 将图片转化为二进制格式
            example = tf.train.Example(
                features=tf.train.Features(feature={
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[class_dict[class_name]])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                    'img_width': tf.train.Feature(int64_list=tf.train.Int64List(value=[size[0]])),
                    'img_height': tf.train.Feature(int64_list=tf.train.Int64List(value=[size[1]])),
                    'img_path':tf.train.Feature(bytes_list=tf.train.BytesList(value=[np.array(os.path.join(data_basepath,
                                                                                                           class_name,image_name)).tobytes()]))
                }))
            writer.write(example.SerializeToString())  # 序列化为字符串
    writer.close()


def read_image_tfrecord(tfrecord_path, image_type):
    features = model.read_tfrecord(tfrecord_path,{
                                       'label': tf.FixedLenFeature([], tf.int64),
                                       'img_raw' : tf.FixedLenFeature([], tf.string),
                                       'img_width': tf.FixedLenFeature([], tf.int64),
                                       'img_height': tf.FixedLenFeature([], tf.int64),
                                   })
    image = tf.decode_raw(features['img_raw'], image_type)
    height = tf.cast(features['img_height'],tf.int32)
    width = tf.cast(features['img_width'],tf.int32)
    label = tf.cast(features['label'],tf.int32)
    return image,height,width,label


def testWriter():
    tfrecord_path = '/media/ai/data/workrooms/datas/tfrecord/flowers'
    type_dict = {'test':'testdata','train':'traindata','val':'valdata'}
    base_path = '/media/ai/data/workrooms/datas/org/flower_photos'
    class_dict = {'daisy':0,'dandelion':1,'roses':2,'sunflowers':3,'tulips':4}
    for dir in type_dict.keys():
        create_image_tfrecord(os.path.join(tfrecord_path,dir),type_dict[dir],
                              os.path.join(base_path,dir),class_dict)


def testReader():
    tfrecord_path = '/media/ai/data/workrooms/datas/tfrecord/flowers'
    type_dict = {'test': 'testdata.tfrecords-00000', 'train': 'traindata.tfrecords-00000', 'val': 'valdata.tfrecords-00000'}
    features = model.read_tfrecord(os.path.join(tfrecord_path,'test',type_dict['test']), {
        'label': tf.FixedLenFeature([], tf.int64),
        'img_raw': tf.FixedLenFeature([], tf.uint8),
        'img_width': tf.FixedLenFeature([], tf.int64),
        'img_height': tf.FixedLenFeature([], tf.int64),
        'img_path': tf.FixedLenFeature([],tf.uint8)
    })

    #image_re = tf.reshape(image,shape=[100,200,3],name='reshape')
    #imagebatch, labelbatch = tf.train.batch([image_re,label],16,num_threads=10,capacity=64)
    with gfile.FastGFile(os.path.join(MODEL_DIR,MODEL_FILE),'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    image_de = tf.decode_raw(features['img_raw'],tf.uint8)
    bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(graph_def,return_elements=[BOTTLENECK_TENSOR_NAME,JPEG_DATA_TENSOR_NAME])
    with tf.Session() as sess:  # 开始一个会话
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
       # sess.run(labelbatch)
        for step in range(1000):
            print('step %d, label is' %(step))
            bottleneck_value = sess.run(bottleneck_tensor, {jpeg_data_tensor: image_de.eval()})
        coord.request_stop()
        coord.join(threads)

