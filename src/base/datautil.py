import tensorflow as tf
import os


def int64_feature(values):
    if not isinstance(values,(tuple,list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(values=values))


def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(values=[values]))


def float_feature(values):
    if not isinstance(values,(tuple,list)):
        values = [values]
    return tf.train.Feature(float_list=tf.train.FloatList(values=values))


def image_classify_encode(base_dir):
    label_list = os.listdir(base_dir)
    label_id = 0
    for label_name in label_list:
        image_data = tf.gfile.FastGFile(image_path, 'rb').read()
        example = tf.train.Example(features=tf.train.Features(feature={
            'image': bytes_feature(image_data),
            'label': int64_feature(label),
        }))


def image_classify_decode(tfrecord_list, batch_size, num_classes,num_epochs=None,channel=3):
    filename_queue = tf.train.string_input_producer(tfrecord_list,num_epochs=num_epochs)
    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)
    image_features = tf.parse_single_example(serialized_example,
                                             features={
                                                 'image': tf.FixedLenFeature([],tf.string),
                                                 'label': tf.FixedLenFeature([],tf.int64)
                                             })
    image_decode = tf.image.decode_image(image_features['image'],channels=channel)
    image_label = image_features['label']
    image_batch, label_batch = tf.train.batch([image_decode,image_label],
                                              batch_size=batch_size,
                                              num_threads=4,
                                              capacity=5*batch_size)
    label_batch = tf.one_hot(label_batch, depth=num_classes, dtype=tf.float32)
    return image_batch, label_batch