import base.layers as layers
import tensorflow as tf
from tensorflow.contrib import slim

def image_classify(input,classnum):
    fc1 = layers.fc_layer('fc_classify',input,500,activate_fun=tf.nn.relu)
    return layers.fc_layer('output',fc1,classnum,activate_fun=None)

def image_classify_cnn(input,classnum):
    conv1 = layers.conv_layer('conv1',input,ksize=[5,5],depth=128,kstep=[1,1],padding='SAME')
    pool2 = layers.pool_layer('pool2',conv1,ksize=[2,2],kstep=[2,2],padding='SAME',pool_fun=tf.nn.max_pool)
    conv3 = layers.conv_layer('conv3',pool2,ksize=[5,5],depth=256,kstep=[1,1],padding='SAME')
    pool4 = layers.pool_layer('pool4',conv3,ksize=[2,2],kstep=[2,2],padding='SAME',pool_fun=tf.nn.max_pool)
    conv5 = layers.conv_layer('conv5',pool4,ksize=[5,5],depth=128,kstep=[1,1],padding='SAME')
    pool6 = layers.pool_layer('pool6',conv5,ksize=[2,2],kstep=[2,2],padding='SAME',pool_fun=tf.nn.max_pool)
    input = layers.convert_n_to_2('convert',pool6)
    return image_classify(input,classnum)