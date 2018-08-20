import base.layers as layers
import tensorflow as tf
from tensorflow.contrib import slim

def image_classify(input,classnum):
    fc1 = layers.fc_layer('fc_classify',input,500,activate_fun=tf.nn.relu)
    return layers.fc_layer('output',fc1,classnum,activate_fun=None)
