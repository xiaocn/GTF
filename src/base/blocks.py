import tensorflow as tf
from base import layers


def classify_features(input_tensor, classnum):
    fc1 = layers.fc_layer('fc_classify', input_tensor, 500, activate_fun=tf.nn.relu)
    return layers.fc_layer('output', fc1, classnum, activate_fun=None)


def extracted_features(input_tensor):
    conv1 = layers.conv_layer('conv1', input_tensor, ksize=[5, 5], depth=128, kstep=[1, 1], padding='SAME')
    pool2 = layers.pool_layer('pool2', conv1, ksize=[2, 2], kstep=[2, 2], padding='SAME', pool_fun=tf.nn.max_pool)
    conv3 = layers.conv_layer('conv3', pool2, ksize=[5, 5], depth=256, kstep=[1, 1], padding='SAME')
    pool4 = layers.pool_layer('pool4', conv3, ksize=[2, 2], kstep=[2, 2], padding='SAME', pool_fun=tf.nn.max_pool)
    conv5 = layers.conv_layer('conv5', pool4, ksize=[5, 5], depth=128, kstep=[1, 1], padding='SAME')
    pool6 = layers.pool_layer('pool6', conv5, ksize=[2, 2], kstep=[2, 2], padding='SAME', pool_fun=tf.nn.max_pool)
    return pool6