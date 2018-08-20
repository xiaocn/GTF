import glob
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

BOTTLENECK_TENSOR_SIZE = 2048

BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'

JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'

MODEL_DIR = '/media/ai/data/workrooms/pre_model/inception_v3'

MODEL_FILE = 'tensorflow_inception_graph.pb'

#CACHE_DIR = '/tmp/bottleneck'

#INPUT_DATA = '/media/ai/data/workrooms/datas/flower_photos'
CACHE_DIR = '/media/ai/data/workrooms/datas/mprh/train_data'

INPUT_DATA = '/media/ai/data/workrooms/datas/classify'
VALIDATION_PERCENTAGE = 10

TEST_PERCENTAGE = 10

LEARNING_RATE = 0.01
STEPS = 4000
BATCH = 100

MODEL_SAVE_PATH = "/media/ai/data/workrooms/models/mprh_model/"
MODEL_NAME = "mprh_model.ckpt"

def create_image_list(testing_percentage, validation_percantage):
    result = {}
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]

    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        extensions = ['jpg','JPG','jpeg', 'JPEG','bmp','BMP','png','PNG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA,dir_name,'*.'+extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list: continue
        label_name = dir_name.lower()
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            change =np.random.randint(100)
            if change < validation_percantage:
                validation_images.append(base_name)
            elif change < (testing_percentage+validation_percantage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)
        result[label_name] = {
            'dir':dir_name,
            'training':training_images,
            'testing':testing_images,
            'validation':validation_images,
        }
    return result


def get_image_path(image_lists, image_dir, label_name, index, category):
    label_lists = image_lists[label_name]
    category_list = label_lists[category]
    mod_index = index % len(category_list)

    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']

    full_path = os.path.join(image_dir,sub_dir,base_name)
    return full_path


def get_bottleneck_path(image_lists, label_name, index, category):
    return get_image_path(image_lists,CACHE_DIR,label_name,index,category)+'.txt'


def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
    bottleneck_value = sess.run(bottleneck_tensor,{image_data_tensor:image_data})
    bottleneck_values = np.squeeze(bottleneck_value)
    return bottleneck_values


def get_or_create_bottleneck(sess, image_lists, label_name, index, category, jpeg_data_tensor, bottleneck_tensor):
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(CACHE_DIR,sub_dir)
    if not os.path.exists(sub_dir_path): os.makedirs(sub_dir_path)
    bottleneck_path = get_bottleneck_path(image_lists,label_name,index,category)
    if not os.path.exists(bottleneck_path):
        image_path = get_image_path(image_lists,INPUT_DATA,label_name,index,category)
        image_data = gfile.FastGFile(image_path,'rb').read()
        bottleneck_values = run_bottleneck_on_image(sess,image_data,jpeg_data_tensor,bottleneck_tensor)
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path,'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)
    else:
        with open(bottleneck_path,'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    return bottleneck_values


def get_random_cached_bottlenecks(sess, n_classes, image_lists, how_many, category, jpeg_data_tensor, bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    for _ in range(how_many):
        label_index = random.randrange(n_classes)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(65536)
        bottleneck = get_or_create_bottleneck(sess,image_lists,label_name,image_index,category,jpeg_data_tensor,bottleneck_tensor)
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)
    return bottlenecks,ground_truths


def get_test_bottlenecks(sess,image_lists,n_classes,jpeg_data_tensor,bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    label_name_list = list(image_lists.keys())
    for label_index,label_name in enumerate(label_name_list):
        categroy = 'testing'
        for index,unused_base_name in enumerate(image_lists[label_name][categroy]):
            bottleneck = get_or_create_bottleneck(sess, image_lists,label_name,index,categroy,jpeg_data_tensor,bottleneck_tensor)
            ground_truth = np.zeros(n_classes, dtype=np.float32)
            ground_truth[label_index] = 1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
    return bottlenecks,ground_truths


def main(agrs=None):
    image_lists = create_image_list(TEST_PERCENTAGE,VALIDATION_PERCENTAGE)
    n_classes = len(image_lists.keys())
    with gfile.FastGFile(os.path.join(MODEL_DIR,MODEL_FILE),'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(graph_def,return_elements=[BOTTLENECK_TENSOR_NAME,JPEG_DATA_TENSOR_NAME])

    bottleneck_input = tf.placeholder(tf.float32,[None,BOTTLENECK_TENSOR_SIZE], name='input')

    ground_truth_input = tf.placeholder(tf.float32,[None,n_classes],name='output')
    with tf.name_scope('final_training_ops'):
        weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE,n_classes],stddev=0.001))
        biases = tf.Variable(tf.zeros([n_classes]))
        logits = tf.matmul(bottleneck_input,weights)+biases
        final_tensor = tf.nn.softmax(logits)
    global_step = tf.Variable(0,trainable=False)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=ground_truth_input)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('loss',cross_entropy_mean)
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy_mean,global_step=global_step)
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(final_tensor,1),tf.argmax(ground_truth_input,1))
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        tf.summary.scalar('accuracy', evaluation_step)
    saver = tf.train.Saver()
    merged_summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('/media/ai/data/workrooms/logs/mprh_log/')
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        summary_writer.add_graph(sess.graph)
        for i in range(STEPS):
            train_bottlenecks, train_ground_truths = get_random_cached_bottlenecks(
                sess,n_classes,image_lists,BATCH,'training',jpeg_data_tensor,bottleneck_tensor)
            step, summary_str = sess.run([train_step, merged_summary_op], feed_dict={bottleneck_input:train_bottlenecks,ground_truth_input:train_ground_truths})
            summary_writer.add_summary(summary_str, global_step=step)
            if i % 100 == 0 or i+1 == STEPS:
                validation_bottlenecks, evaluation_ground_truth = get_random_cached_bottlenecks(
                    sess,n_classes,image_lists,BATCH,'validation',jpeg_data_tensor,bottleneck_tensor)
                evaluation_accuracy = sess.run(evaluation_step,feed_dict={
                    bottleneck_input:validation_bottlenecks,
                    ground_truth_input:evaluation_ground_truth})
                print('%d step, %d examples accuracy is %.1f%%' % (i,BATCH,evaluation_accuracy*100))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))
        test_bottlenecks, test_ground_truth = get_test_bottlenecks(sess, image_lists, n_classes, jpeg_data_tensor, bottleneck_tensor)
        test_accuracy = sess.run(evaluation_step, feed_dict={
                bottleneck_input:test_bottlenecks,
                ground_truth_input:test_ground_truth
            })
        print('final test accuracy is %.lf%%' % (test_accuracy * 100))
        graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["output"])
        tf.train.write_graph(graph, '/media/ai/data/workrooms/models/mprh_model/', '/media/ai/data/workrooms/models/pb_models/mprh_model/mprh_graph.pb', as_text=False)


if __name__  ==  '__main__':
    tf.app.run()