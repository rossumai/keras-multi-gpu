from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, Input
import os
import tensorflow as tf

from data_parallel import DataParallelModel, DataParallelOptimizer

def load_data():
    # input image dimensions
    img_rows, img_cols = 28, 28
    num_classes = 10

    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1).astype('float32') / 255
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1).astype('float32') / 255

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return x_train, y_train, x_test, y_test

def make_basic_model(input_shape, num_classes):
    input = Input(shape=input_shape)
    x = Conv2D(16, 3, activation='selu')(input)
    x = Dropout(0.3)(x)
    x = Conv2D(32, 3, activation='selu')(x)
    x = Dropout(0.3)(x)
    x = Conv2D(64, 3, activation='selu')(x)
    x = Dropout(0.3)(x)
    x = Flatten()(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input, outputs=output)

    model.summary()

    return model

def train_single_gpu(batch_size=256, epochs=10):
    x_train, y_train, x_test, y_test = load_data()
    input_shape = x_train.shape[1:]
    num_classes = y_train.shape[-1]

    model = make_basic_model(input_shape, num_classes)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))

    # score = model.evaluate(x_test, y_test, verbose=0)
    # print('Test loss:', score[0], 'accuracy:', score[1])

def train_multi_gpu(subbatch_size=256, epochs=10, gpus=2):
    x_train, y_train, x_test, y_test = load_data()
    input_shape = x_train.shape[1:]
    num_classes = y_train.shape[-1]
    gpu_count = gpu_count = len([dev for dev in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if len(dev.strip()) > 0])
    batch_size = subbatch_size * gpus

    print('CUDA_VISIBLE_DEVICES', os.environ.get('CUDA_VISIBLE_DEVICES', ''), 'gpu_count:', gpu_count)

    basic_model = make_basic_model(input_shape, num_classes)
    parallel_model = DataParallelModel.create(basic_model, gpu_count)

    parallel_model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])

    parallel_model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))

    # score = model.evaluate(x_test, y_test, verbose=0)
    # print('Test loss:', score[0], 'accuracy:', score[1])


if __name__ == '__main__':
    train_multi_gpu()

## Basic model
# 333,066 params
# batch_size=128 -> 9s / epoch
# batch_size=256 -> 7.5s / epoch
