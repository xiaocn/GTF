import tensorflow as tf
import os


def read_tfrecord(data_path,feature_dict):
    data_files = tf.gfile.Glob(data_path)
    filename_queue = tf.train.string_input_producer(data_files,shuffle=True)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,features=feature_dict)
    return features


def model_test(modelpath, data_dict, output_list):
    with tf.Graph().as_default():
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(modelpath, 'rb') as f:
            serialized_graph = f.read()
            graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(graph_def,name='')
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            input_dict = {}
            for input_name in data_dict.keys():
                temp_tensor = sess.graph.get_tensor_by_name(input_name)
                input_dict[temp_tensor] = data_dict[input_name]
            output_tensor = []
            for output_name in output_list:
                temp_tensor = sess.graph.get_tensor_by_name(output_name)
                output_tensor.append(temp_tensor)
            result_output = sess.run(output_tensor, feed_dict=input_dict)
    return result_output


def model_eval(input_dict, output_list,modelpath, moving_average_decay=0):
    result_output = None
    with tf.Graph().as_default():
        saver = tf.train.Saver()
        if moving_average_decay > 0:
            variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay)
            variable_to_restore = variable_averages.variables_to_restore()
            saver = tf.train.Saver(variable_to_restore)
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(modelpath)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess,ckpt.model_checkpoint_path)
                result_output = sess.run(output_list,feed_dict=input_dict)
    return result_output


def train(save_model_path, logit, label, input_dict, learning_rate,
          max_save_num=5,
          save_step=200,
          message_step=1000,
          train_steps=30000,
          moving_average_decay=0,
          optimizer=tf.train.GradientDescentOptimizer,
          loss_fun=tf.nn.softmax_cross_entropy_with_logits):
    global_step = tf.Variable(0, trainable=False)
    cross_entropy = loss_fun(logits=logit, labels=label)
    accuracy_prediction = tf.equal(tf.argmax(logit,1),tf.argmax(label,1))
    accuracy = tf.reduce_mean(tf.cast(accuracy_prediction, tf.float32))
    loss = tf.reduce_mean(cross_entropy)
    train_step = optimizer(learning_rate).minimize(loss, global_step=global_step)
    dependen_lsit = [train_step]
    if moving_average_decay > 0:
        variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay, global_step)
        variable_averages_op = variable_averages.apply(tf.trainable_variables())
        dependen_lsit.append(variable_averages_op)
    with tf.control_dependencies(dependen_lsit):
        train_op = tf.no_op(name='train')
    saver = tf.train.Saver(max_to_keep=max_save_num)
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        for i in range(train_steps):
            _, loss_value, accuracy_value,step = sess.run([train_op,loss,accuracy,global_step], feed_dict=input_dict)
            if i % message_step == 0:
                print('%d step, loss is %.2f, accuracy is %.2f%%' % (step, loss_value, accuracy_value*100))
                saver.save(sess, save_model_path, global_step=global_step)
            if i % save_step == 0:
                metadir = os.path.join(save_model_path,'meta')
                if not os.path.exists(metadir):os.makedirs(metadir)
                saver.save(sess,os.path.join(metadir,'model.ckpt'))


from tensorflow.examples.tutorials.mnist import input_data
from base import blocks
if __name__=='__main__':
    x = tf.placeholder(tf.float32,[None,784],name='input')
    y_ = tf.placeholder(tf.float32,[None,10],name='output')
    mnist = input_data.read_data_sets("/media/ai/data/workrooms/datas/org/mnist", one_hot=True)
    logits = blocks.classify_features(x,10)
    xs,ys =mnist.train.next_batch(32)
    train("/media/ai/data/workrooms/models/mnist_model",logits,y_,{x : xs, y_: ys},0.8)