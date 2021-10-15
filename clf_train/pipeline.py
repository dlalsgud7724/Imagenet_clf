import numpy as np
import tensorflow as tf
import os
import pandas as pd
import sys

sys.path.append(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))

from parameters import tfrecord_parameter as PARAM
class input_pipeline(object):
    def __init__(self):
        pass
    def decode_fun(self, serialized_example):
        image_feature_description = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'depth': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string)
        }

        parsed_data = tf.io.parse_single_example(serialized_example,image_feature_description)
        image_raw =tf.io.decode_jpeg( parsed_data['image_raw'],channels=3)
        image_raw = tf.image.resize(image_raw, [224,224])
        label = parsed_data['label']

        return image_raw, label


    def gen_dataset(self, name = 'train'):
        tfrecord_dir = PARAM.TFRECORD_DIR
        tfrecord_list = [os.path.join(tfrecord_dir,file) for file in os.listdir(tfrecord_dir) if file.startswith(name)]
        file_ds = tf.data.Dataset.from_tensor_slices(tfrecord_list)
        ds = file_ds.interleave(tf.data.TFRecordDataset,num_parallel_calls=tf.data.experimental.AUTOTUNE).map(self.decode_fun).prefetch(tf.data.experimental.AUTOTUNE)
        ds = ds.shuffle(buffer_size=1)
        ds = ds.batch(32)
        return ds

