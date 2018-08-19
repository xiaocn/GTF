import tensorflow as tf
import nets.classify_net as classify
import os
#from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from PIL import Image

INPUT_NODE = 784
OUTPUT_NODE = 10

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH = "/media/ai/data/workrooms/models/mnist_model/"
MODEL_NAME = "mnist_model.ckpt"

def create_record():
    basepath = '/media/ai/data/workrooms/datas/classify'
    class_dict = {'clear':0,'muddy':1}
    data_list = []
    for class_name in class_dict.keys():
        for image_name in os.listdir(os.path.join(basepath,class_name)):
            data_list.append({'image_path':os.path.join(basepath,class_name,image_name),
                              'label_name':class_name,
                              'label_index':class_dict[class_name]})

    np.random.shuffle(data_list)
    writer = tf.python_io.TFRecordWriter('/media/ai/data/workrooms/datas/mprh/train.tfrecord')
    for data in data_list:
        #print(data['label_index'])
        label = data['label_index']
        img = Image.open(data['image_path'])
        img = img.resize((70,70),Image.NEAREST)
        data_example = tf.train.Example(features=tf.train.Features(feature={
            "label":tf.train.Feature(float_list=tf.train.FloatList(value=[label])),
            "image":tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tobytes()]))
        }))
        writer.write(data_example.SerializeToString())
    writer.close()

def read_record(tfrecord_file, batch_size):
    filename_queue = tf.train.string_input_producer([tfrecord_file])
    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)
    img_features = tf.parse_single_example(serialized_example,features={
        'label':tf.FixedLenFeature([],tf.int64),
        'image':tf.FixedLenFeature([],tf.string)
    })
    image = tf.decode_raw(img_features['image'],tf.uint8)
    image = tf.reshape(image,[1470])
    label = tf.cast(img_features['label'],tf.int32)
    image_batch,label_batch = tf.train.batch([image,label],
                                             batch_size=batch_size,
                                             num_threads=64,
                                             capacity=2000)
    return image_batch,tf.reshape(label_batch,[batch_size])

def train():
    x = tf.placeholder(tf.float32,[None,1470],name='input')
    y_ = tf.placeholder(tf.float32,[None,OUTPUT_NODE],name='output')

    y = classify.image_classify(x,OUTPUT_NODE)
    global_step = tf.Variable(0,trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_,1))
    loss = tf.reduce_mean(cross_entropy)
   # learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
   #                                            global_step,
   #                                            mnist.train.num_examples/BATCH_SIZE,
   #                                            LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE_BASE).minimize(loss,global_step=global_step)
    with tf.control_dependencies([train_step,variable_averages_op]):
        train_op = tf.no_op(name='train')
    saver = tf.train.Saver()
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(TRAINING_STEPS):
            xs, ys = read_record('/media/ai/data/workrooms/datas/mprh/train.tfrecord',32)
            _,loss_value,step = sess.run([train_op,loss,global_step],feed_dict={x:xs,y_:ys})
            if i % 1000 == 0:
                print('%d step, loss is %g.' % (step,loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)


def main(avg=None):
   # mnist = input_data.read_data_sets("/media/ai/data/workrooms/datas",one_hot=True)
    train()


if __name__ == '__main__':
    create_record()
   # main()