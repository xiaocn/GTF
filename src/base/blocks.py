import base.layers as layers
import tensorflow as tf


def stem_block(name, input):
    with tf.name_scope(name) as scope:
        conv_1 = layers.conv_layer(scope+'conv_1',input,ksize=[3,3],depth=32,kstep=[2,2],padding='VALID')
        conv_2 = layers.conv_layer(scope+'conv_2',conv_1,ksize=[3,3],depth=32,kstep=[1,1],padding='VALID')
        conv_3 = layers.conv_layer(scope+'conv_3',conv_2,ksize=[3,3],depth=64,kstep=[1,1],padding='SAME')

        pool_1b1 = layers.pool_layer(scope+'pool_1b1',conv_3,ksize=[3,3],kstep=[2,2],padding='VALID',pool_fun=tf.nn.max_pool)
        conv_4b2 = layers.conv_layer(scope+'conv_4b2',conv_3,ksize=[3,3],depth=96, kstep=[2,2],padding='VALID')
        concat_0 = tf.concat(3,[pool_1b1,conv_4b2],name=scope+'concat_0')

        conv_5b3 = layers.conv_layer(scope+'conv_5b3',concat_0,ksize=[1,1],depth=64,kstep=[1,1],padding='SAME')
        conv_6b3 = layers.conv_layer(scope+'conv_6b3',conv_5b3,ksize=[3,3],depth=96,kstep=[1,1],padding='VALID')

        conv_7b4 = layers.conv_layer(scope+'conv_7b4',concat_0,ksize=[1,1],depth=64,kstep=[1,1],padding='SAME')
        conv_8b4 = layers.conv_layer(scope+'conv_8b4',conv_7b4,ksize=[7,1],depth=64,kstep=[1,1],padding='SAME')
        conv_9b4 = layers.conv_layer(scope+'conv_9b4', conv_8b4,ksize=[1,7],depth=64,kstep=[1,1],padding='SAME')
        conv_10b4 = layers.conv_layer(scope+'conv_10b4',conv_9b4,ksize=[3,3],depth=96,kstep=[1,1],padding='VALID')
        concat_1 = tf.concat(3,[conv_6b3,conv_10b4],name=scope+'concat_1')

        conv_11b5 = layers.conv_layer(scope+'conv_11b5', concat_1, ksize=[3,3],depth=192,kstep=[1.1],padding='VALID')
        pool_2b6 = layers.pool_layer(scope+'pool_2b6',concat_1,ksize=[1,1],kstep=[2,2],padding='VALID',pool_fun=tf.nn.max_pool)
        concat_2 = tf.concat(3,[conv_11b5,pool_2b6],name=scope+'concat_2')

        return concat_2


def inception_A_block(name, input):
    with tf.name_scope(name) as scope:
        pool_1b1 = layers.pool_layer(scope+'pool_1b1',input,ksize=[1,1],kstep=[1,1],padding='SAME',pool_fun=tf.nn.avg_pool)
        conv_1b1 = layers.conv_layer(scope+'conv_1b1',pool_1b1,ksize=[1,1],depth=96,kstep=[1,1],padding='SAME')

        conv_2b2 = layers.conv_layer(scope+'conv_2b2',input,ksize=[1,1],depth=96,kstep=[1,1],padding='SAME')

        conv_3b3 = layers.conv_layer(scope+'conv_3b3',input,ksize=[1,1],depth=64,kstep=[1,1],padding='SAME')
        conv_4b3 = layers.conv_layer(scope+'conv_4b3', conv_3b3, ksize=[3,3],depth=96,kstep=[1,1],padding='SAME')

        conv_5b4 = layers.conv_layer(scope+'conv_5b4',input,ksize=[1,1],depth=64,kstep=[1,1],padding='SAME')
        conv_6b4 = layers.conv_layer(scope+"conv_6b4",conv_5b4,ksize=[3,3],depth=96,kstep=[1,1],padding='SAME')
        conv_7b4 = layers.conv_layer(scope+'conv_7b4', conv_6b4,ksize=[3,3],depth=96,kstep=[1,1],padding='SAME')

        concat_0 = tf.concat(3,[conv_1b1,conv_2b2,conv_4b3,conv_7b4],name=scope+'concat_0')

        return concat_0

def reduction_A_block(name, input):
    with tf.name_scope(name) as scope:
        pool_1b1 = layers.pool_layer(scope+'pool_1b1',input,ksize=[3,3],kstep=[2,2],padding='VALID',pool_fun=tf.nn.max_pool)

        conv_1b2 = layers.conv_layer(scope+'conv_1b2',input,ksize=[3,3],depth=384,kstep=[2,2],padding='VALID')

        conv_2b3 = layers.conv_layer(scope+'conv_2b3',input,ksize=[1,1],depth=192,kstep=[1,1],padding='SAME')
        conv_3b3 = layers.conv_layer(scope+'conv_3b3',conv_2b3,ksize=[3,3],depth=224,kstep=[1,1],padding='SAME')
        conv_4b3 = layers.conv_layer(scope+'conv_4b3',conv_3b3,ksize=[3,3],depth=256,kstep=[2,2],padding='VALID')

        concat_0 = tf.concat(3,[pool_1b1,conv_1b2,conv_4b3],name=scope+'concat_0')

        return concat_0

