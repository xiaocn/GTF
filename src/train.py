import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer('train_step',30000,'训练的步数间隔')
flags.DEFINE_integer('max_save_num',5,'保存meta文件的最大数目')
flags.DEFINE_integer('message_step',1000,'打印训练信息的步数间隔')
flags.DEFINE_integer('save_step',200,'保存meta模型的部属间隔')



def main(_):
    x = tf.placeholder(tf.float32, [None, 784], name='input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='output')

    with tf.name_scope('layer') as scope:
        w_layer = tf.get_variable(scope+'w',[784,500],tf.float32)
        b_layer = tf.get_variable(scope+'b', [500], tf.float32)
    layer = tf.nn.relu(tf.matmul(x,w_layer)+b_layer)

    with tf.name_scope('final') as scope:
        w_output = tf.get_variable(scope+'w',[500,10],tf.float32)
        b_output = tf.get_variable(scope+'b',[10], tf.float32)
    y = tf.matmul(layer,w_output)+b_output
    loss = tf.nn.softmax_cross_entropy_with_logits(y, y_)
    loss_mean = tf.reduce_mean(loss)
    tf


if __name__=='__main__':
    tf.app.run()
