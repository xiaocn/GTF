import tensorflow as tf


def read_tfrecord(filename,feature_dict):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,features=feature_dict)
    return features


def load_model(modelpath,name):
    graph = tf.Graph()
    with graph.as_default():
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(modelpath,'rb') as f:
            serialized_graph = f.read()
            graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(graph_def,name=name)
    return graph


def eval(moving_average_decay,input_dict, y,y_,modelpath):
    with tf.Graph().as_default() as g:
        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay)
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(modelpath)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess,ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                accuracy_score = sess.run(accuracy,feed_dict=input_dict)
                print("%s step, accuracy is %g." % (global_step,accuracy_score))
            else:
                print('no model')


def train(train_steps, moving_average_decay, learning_rate,savepath,input_dict):
    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    loss = tf.reduce_mean(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')
    saver = tf.train.Saver()
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            for epoch_index in range(train_steps):
                tra_images, tra_labels = sess.run([tra_image_batch, tra_label_batch])
                accuracy, mean_cost_in_batch, return_correct_times_in_batch, _ = sess.run(
                    [graph['accuracy'], graph['cost'], graph['correct_times_in_batch'], graph['optimize']], feed_dict={
                        graph['x']: tra_images,
                        graph['lr']: learning_rate,
                        graph['y']: tra_labels
                    })
                if epoch_index % epoch_delta == 0:
                    # 开始在 train set上计算一下accuracy和cost
                    print("index[%s]".center(50, '-') % epoch_index)
                    print("Train: cost_in_batch：{},correct_in_batch：{},accuracy：{}".format(mean_cost_in_batch,
                                                                                           return_correct_times_in_batch,
                                                                                           accuracy))

                    # 开始在 test set上计算一下accuracy和cost
                    val_images, val_labels = sess.run([val_image_batch, val_label_batch])
                    mean_cost_in_batch, return_correct_times_in_batch = sess.run(
                        [graph['cost'], graph['correct_times_in_batch']], feed_dict={
                            graph['x']: val_images,
                            graph['y']: val_labels
                        })
                    print("***Val: cost_in_batch：{},correct_in_batch：{},accuracy：{}".format(mean_cost_in_batch,
                                                                                            return_correct_times_in_batch,
                                                                                            return_correct_times_in_batch / batch_size))

                if epoch_index % 500 == 0:
                    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["output"])
                    with tf.gfile.FastGFile(pb_file_path, mode='wb') as f:
                        f.write(constant_graph.SerializeToString())
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
        coord.join(threads)
        sess.close()