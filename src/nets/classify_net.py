import base.layers as layers
import tensorflow as tf
from tensorflow.contrib import slim

def image_classify(input,classnum):
    fc1 = layers.fc_layer('fc_classify',input,500,activate_fun=tf.nn.relu)
    return layers.fc_layer('output',fc1,classnum,activate_fun=None)

def classify_slim(image,label):
    net = slim.conv2d(image, 128, [5, 5], scope='conv1')
    net = slim.max_pool2d(net, [2, 2], scope='pool1')
    net = slim.conv2d(net, 256, [5, 5], scope='conv2')
    net = slim.max_pool2d(net, [2, 2], scope='pool2')
    net = slim.flatten(net, scope='flatten')
    net = slim.fully_connected(net, 128, scope='fully_connected1')
    logits = slim.fully_connected(net, 2, activation_fn=None, scope='fully_connected2')

   # prob = slim.softmax(logits)
    loss = slim.losses.softmax_cross_entropy(logits, label)

    train_op = slim.optimize_loss(loss, slim.get_global_step(),
                                  learning_rate=0.0001,
                                  optimizer='Adam')

    return train_op