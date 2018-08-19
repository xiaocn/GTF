import nets.classify_net as classify
import tensorflow as tf




def read_tfrecord(filename):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.uint8),
            'label': tf.FixedLenFeature([], tf.float32)
        })

    image = tf.decode_raw(features['image'], tf.float32)
    label = tf.decode_raw(features['label'], tf.float32)

    image = tf.reshape(image, [70, 70, 3])
    label = tf.reshape(label, [2])

    image, label = tf.train.batch([image, label],batch_size=10,capacity=10)

    return image, label


def main():
    train_images, train_labels = read_tfrecord('/media/ai/data/workrooms/datas/mprh/train.tfrecord')
    train_op = classify.classify_slim(train_images, train_labels)

    step = 0
    with tf.Session() as sess:
        init_op = tf.group(
            tf.local_variables_initializer(),
            tf.global_variables_initializer())
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        while step < 3000:
            sess.run([train_op])

            if step % 100 == 0:
                print('step: {}'.format(step))

            step += 1

        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    main()