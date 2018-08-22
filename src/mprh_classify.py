import models.classify_model as classify
import tensorflow as tf
import models.base_model as model


def train():
    image, height, width, label = classify.read_image_tfrecord('')
    tf.train.batch()