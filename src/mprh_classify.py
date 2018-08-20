import nets.classify_net as classify
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from PIL import Image
import base.datatools as datatool
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
import os

MODEL_SAVE_PATH = "/media/ai/data/workrooms/models/mnist_model/"
MODEL_NAME = "mnist_model.ckpt"


def read_tfrecord(filename):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.float32),
            'label': tf.FixedLenFeature([], tf.float32)
        })

    image = tf.decode_raw(features['image'], tf.float32)
    label = tf.decode_raw(features['label'], tf.float32)

    image = tf.reshape(image, [70, 70, 3])
    label = tf.reshape(label, [2])

    image, label = tf.train.batch([image, label],batch_size=2,capacity=10)

    return image, label


def main():
    #mnist = input_data.read_data_sets("/media/ai/data/workrooms/datas", one_hot=True)
    basepath = '/media/ai/data/workrooms/datas/classify'
    class_dict = {'clear': 0, 'muddy': 1}
    datasets = datatool.get_pathlist(basepath, class_dict)
    datalen = len(datasets)
    x = tf.placeholder(dtype=tf.float32,shape=[1, 1200],name='input')
    y_ = tf.placeholder(dtype=tf.float32,shape=[1, 2], name='output')
    y = classify.image_classify(x, 2)
    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(0.99, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    #loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    #loss = tf.reduce_mean(cross_entropy)
    loss = tf.reduce_mean
    learning_rate = tf.train.exponential_decay(0.8,
                                               global_step,
                                               datalen,
                                               0.99)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')
    saver = tf.train.Saver()
    init_op = tf.global_variables_initializer()
    config = tf.ConfigProto(device_count={'gpu': 0})
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        for i in range(TRAINING_STEPS):
            data = datasets[int(np.random.random() * datalen)]
            image = Image.open(data['image_path'])
            #print(image.mode)
            image = image.resize((20, 20), Image.NEAREST)
            #print(image)
            im = np.array(image)
            #print(im.shape)
            imageArray = np.reshape(im, [1,1200])
            label = np.zeros([1,2])
            if [data['label_index']]==0:
                label[0][0] = 1
            else:
                label[0][1] = 1
            _,loss_value,step = sess.run([train_op,loss,global_step],feed_dict={x:imageArray,y_:label})
            if i % 1000 == 0:
                print('%d step, loss is %g.' % (step,loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)

if __name__ == '__main__':
    main()