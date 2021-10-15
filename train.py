import numpy as np
import tensorflow as tf
import os
import pandas as pd

from clf_train.pipeline import input_pipeline
from classifiers.Resnet34 import Resnet34

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

if __name__ == "__main__":
    pipeline = input_pipeline()
    train_ds = pipeline.gen_dataset('train')
    val_ds = pipeline.gen_dataset('val')

    model = Resnet34(1000)
    model.build((32,224,244,3))
    model.compile(optimizer='adam',
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['sparse_categorical_accuracy'])
    model.fit(train_ds, epochs = 100)
