from numpy.core.numeric import True_
from pandas.core.accessor import PandasDelegate
import tensorflow as tf
import os
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm
import random
import io
from threading import Thread

from parameters import tfrecord_parameter as PARAM


def _int64_feature(value):
    #   """Wrapper for inserting int64 features into Example proto."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
#   """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
#   """Wrapper for inserting float features into Example proto."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _parse_image_function(example_proto):
    image_feature_description = {
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'depth': tf.io.FixedLenFeature([], tf.int64),
    'label': tf.io.FixedLenFeature([], tf.int64),
    'image_raw': tf.io.FixedLenFeature([], tf.string)
    }

    return tf.io.parse_single_example(example_proto, image_feature_description)

def image_to_byte_array(image:Image):
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format=image.format)
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr


def _convert_example(image_string, label, width, height, depth):
    feature = {
        'image_raw' : _bytes_feature(image_string),
        'label' : _int64_feature(label),
        'width' : _int64_feature(width),
        'height' : _int64_feature(height),
        'depth' : _int64_feature(depth)
    }
    serialized_example = tf.train.Example(features = tf.train.Features(feature = feature))
    return serialized_example

def convert_tfrecord(label_df, image_dir, name = 'val', thread_num = 0):
    if name == 'train':
        dataset_index = [int(len(label_df) / PARAM.NUM_PROCESSOR) * i for i in range(PARAM.NUM_PROCESSOR + 1)]
        label_df = label_df.iloc[dataset_index[thread_num] : dataset_index[thread_num + 1]].reset_index(drop = True)
    record_file = os.path.join(PARAM.TFRECORD_DIR, name + '_' + str(thread_num) + '.tfrecords')

    with tf.io.TFRecordWriter(record_file) as writer:
        for i in tqdm(range(len(label_df))):
            if name == 'val':
                with open(os.path.join(image_dir,label_df.iloc[i]['ImageId'] + '.JPEG'), 'rb') as f:
                    image_string = f.read()
                    image = tf.io.decode_jpeg(image_string, channels=3)
            
            if name == 'train':
                with open(os.path.join(image_dir, label_df.iloc[i]['PredictionString'], label_df.iloc[i]['ImageId']), 'rb') as f:
                    image_string = f.read()
                    image = tf.io.decode_jpeg(image_string, channels=3)

            label = int(label_df.iloc[i]['class'])
            width = image.shape[0]
            height = image.shape[1]
            depth = image.shape[2]
            tf_example = _convert_example(image_string, label, width, height, depth)
            writer.write(tf_example.SerializeToString())

def convert_validation():
    image_dir = PARAM.RAW_VAL_IMAGE_DIR
    labal_dir = PARAM.RAW_VAL_LABEL_DIR

    # Read csv file include label of validation dataset
    label_df = pd.read_csv(labal_dir)

    # Convert label(string) to int
    label_df['class'] = ''
    for i in tqdm(range(len(label_df))):
        label_df.at[i,'class'] = label_df.iloc[i]['PredictionString'].split(' ')[0]
    label_list = label_df.drop_duplicates('class').sort_values('class').reset_index(drop = True)['class']
    for i in tqdm(range(len(label_df))):
        label_df.at[i,'class'] = label_list[label_list == label_df.iloc[i]['class']].index[0]
    
    # convert tfrecord from the dataset(JPEG)
    convert_tfrecord(label_df, image_dir, name = 'val', thread_num = 0)


def convert_train():
    image_dir = PARAM.RAW_TRAIN_IMAGE_DIR
    labal_dir = PARAM.RAW_TRAIN_LABEL_DIR
    
    # Make empty dataframe for train set label
    label_df = pd.DataFrame()
    
    # label the train set with thier directory
    labellist = os.listdir(image_dir)
    for i, label in tqdm(enumerate(labellist)):
        temp_df = pd.DataFrame()
        path = os.path.join(image_dir,label)
        imagelist = os.listdir(path)
        temp_df['ImageId'] = imagelist
        temp_df['PredictionString'] = label
        temp_df['class'] = i
        label_df = label_df.append(temp_df,ignore_index=True)
    
    # Shuffle the dataset
    label_df = label_df.sample(frac = 1).reset_index(drop = True)

    # Use multithread for convert tfrecord
    thread_list = []
    for i in range(PARAM.NUM_PROCESSOR):
        args = (label_df, image_dir,  'train', i)
        thread_list.append(Thread(target=convert_tfrecord, args=args))
    
    for i in range(PARAM.NUM_PROCESSOR):
        thread_list[i].start()

    for i in range(PARAM.NUM_PROCESSOR):
        thread_list[i].join()

if __name__ == "__main__":

    convert_validation()
    convert_train()
