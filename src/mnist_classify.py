import tensorflow as tf
import nets.classify_net as classify
import os
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE = 784
OUTPUT_NODE = 10

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH = "/media/ai/data/workrooms/models/mnist_model/"
MODEL_NAME = "mnist_model.ckpt"


def eval(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32,[None,INPUT_NODE],name='input')
        y_ = tf.placeholder(tf.float32,[None,OUTPUT_NODE],name='output')
        y = classify.image_classify(x,OUTPUT_NODE)

        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess,ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                accuracy_score = sess.run(accuracy,feed_dict={
                    x:mnist.validation.images,
                    y_:mnist.validation.labels})
                print("%s step, accuracy is %g." % (global_step,accuracy_score))
            else:
                print('no model')

def train(mnist):
    x = tf.placeholder(tf.float32,[None,784],name='input')
    y_ = tf.placeholder(tf.float32,[None,OUTPUT_NODE],name='output')

    y = classify.image_classify(x,OUTPUT_NODE)
    global_step = tf.Variable(0,trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_,1))
    loss = tf.reduce_mean(cross_entropy)
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                              global_step,
                                              mnist.train.num_examples/BATCH_SIZE,
                                              LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    with tf.control_dependencies([train_step,variable_averages_op]):
        train_op = tf.no_op(name='train')
    saver = tf.train.Saver()
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _,loss_value,step = sess.run([train_op,loss,global_step],feed_dict={x:xs,y_:ys})
            if i % 1000 == 0:
                print('%d step, loss is %g.' % (step,loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)


def main(avg=None):
    mnist = input_data.read_data_sets("/media/ai/data/workrooms/datas",one_hot=True)
    #train(mnist)
    eval(mnist)


if __name__ == '__main__':
    main()