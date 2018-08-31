import glob
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from tqdm import tqdm
from PIL import Image
from nets import classify_net as classify
from base import layers

BOTTLENECK_TENSOR_SIZE = 2048

BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'

JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'

MODEL_DIR = '/media/ai/data/workrooms/pre_model/inception_v3'

MODEL_FILE = 'tensorflow_inception_graph.pb'

CACHE_DIR = '/tmp/bottleneck'

MODEL_SAVE_PATH = "/media/ai/data/workrooms/models/mprh_model/"
MODEL_NAME = "mprh_model.ckpt"

train_dir = '/media/ai/data/workrooms/datas/org/classify/train'
val_dir = '/media/ai/data/workrooms/datas/org/classify/val'
test_dir = '/media/ai/data/workrooms/datas/org/classify/test'
class_list= ['clear','muddy']

def get_random_data(path):
    class_index = np.random.randint(2)
    labels = np.zeros([1, 2], dtype=np.float32)
    labels[0][class_index] = 1.0
    image_name_list = os.listdir(os.path.join(path,class_list[class_index]))
    image_name = image_name_list[np.random.randint(len(image_name_list))]
    image_data = gfile.FastGFile(os.path.join(path,class_list[class_index],image_name), 'rb').read()
    return image_data,labels


def train():
    ground_truth_input = tf.placeholder(tf.float32, [None, 2], name='output')
    with gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(graph_def, return_elements=[BOTTLENECK_TENSOR_NAME,
                                                                                          JPEG_DATA_TENSOR_NAME])
    #logits = layers.fc_layer('final_train',bottleneck_tensor,2,activate_fun=None)
    with tf.name_scope('final_training_ops'):
        weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, 2], stddev=0.001))
        biases = tf.Variable(tf.zeros([2]))
        logits = tf.matmul(bottleneck_tensor, weights) + biases
        final_tensor = tf.nn.softmax(logits)
        print(final_tensor)
    global_step = tf.Variable(0, trainable=False)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=ground_truth_input)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('loss', cross_entropy_mean)
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy_mean, global_step=global_step)
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(final_tensor, 1), tf.argmax(ground_truth_input, 1))
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', evaluation_step)
        print(evaluation_step)
    saver = tf.train.Saver(max_to_keep=5)
    merged_summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('/media/ai/data/workrooms/logs/mprh_log/')
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        summary_writer.add_graph(sess.graph)
        for i in range(4000):
            image_data,image_label = get_random_data(train_dir)
            step, summary_str = sess.run([train_step, merged_summary_op],
                                         feed_dict={jpeg_data_tensor: image_data,
                                                    ground_truth_input: image_label})
            summary_writer.add_summary(summary_str, global_step=step)
            if i % 100 == 0 or i + 1 == 4000:
                image_data, image_label = get_random_data(val_dir)
                evaluation_accuracy = sess.run(evaluation_step, feed_dict={
                   jpeg_data_tensor : image_data,
                    ground_truth_input: image_label})
                print('%d step, accuracy is %.1f%%' % (i, evaluation_accuracy * 100))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))
        image_data, image_label = get_random_data(test_dir)
        test_accuracy = sess.run(evaluation_step, feed_dict={
            jpeg_data_tensor: image_data,
            ground_truth_input: image_label
        })
        print('final test accuracy is %.lf%%' % (test_accuracy * 100))
        graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["output"])
        tf.train.write_graph(graph, '/media/ai/data/workrooms/models/mprh_model/', '/media/ai/data/workrooms/models/pb_models/mprh_model/mprh_graph.pb', as_text=False)

