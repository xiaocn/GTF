import tensorflow as tf
import os
from tqdm import tqdm
from PIL import Image
from models import base_model as model
import numpy as np

def create_image_tfrecord():
    tfrecord_base_path = '/media/ai/data/workrooms/datas/tfrecord/flowers'
    type_dict = {'test': 'testdata', 'train': 'traindata', 'val': 'valdata'}
    org_base_path = '/media/ai/data/workrooms/datas/org/flower_photos'
    class_dict = {'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflowers': 3, 'tulips': 4}
    bestnum = 1000

    for dir in type_dict.keys():
        num = 0
        recordfilenum = 0
        tfrecord_path = os.path.join(tfrecord_base_path, dir)
        type_name = type_dict[dir]
        data_basepath =  os.path.join(org_base_path, dir)
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
              #  print(class_dict[class_name])
                example = tf.train.Example(features=tf.train.Features(feature={
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[class_dict[class_name]])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                    'img_width': tf.train.Feature(int64_list=tf.train.Int64List(value=[size[0]])),
                    'img_height': tf.train.Feature(int64_list=tf.train.Int64List(value=[size[1]])),
                    'img_name':tf.train.Feature(bytes_list=tf.train.BytesList(value=[np.array(image_name).tobytes()]))
                }))
                writer.write(example.SerializeToString())  # 序列化为字符串
        writer.close()



def reader():
    tfrecord_path = '/media/ai/data/workrooms/datas/tfrecord/flowers'
    type_dict = {'test': 'testdata.tfrecords-00000', 'train': 'traindata.tfrecords-00000', 'val': 'valdata.tfrecords-00000'}
    features = model.read_tfrecord(os.path.join(tfrecord_path,'test',type_dict['test']), {
        'label': tf.FixedLenFeature([], tf.int64),
        'img_raw': tf.FixedLenFeature([], tf.string),
        'img_width': tf.FixedLenFeature([], tf.int64),
        'img_height': tf.FixedLenFeature([], tf.int64),
        'img_name':tf.FixedLenFeature([],tf.string)
    })
    #image = tf.image.decode_image(features['img_raw'],3)
    label = tf.cast(features['label'],tf.int32)
    #image_re = tf.reshape(image,shape=[100,200,3],name='reshape')
    labelbatch,batch_image_name = tf.train.batch([label,features['img_name']],16,num_threads=10,capacity=64)
    with tf.Session() as sess:  # 开始一个会话
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord=coord)
        label_value,batch_image_value = sess.run([labelbatch,batch_image_name])
        for step in range(1000):
            print('step %d, label is %s, image_name is %s' %(step,'{}'.format(label_value),'{}'.format(batch_image_value)))
        coord.request_stop()
        coord.join(threads)


if __name__ == "__main__":
    reader()