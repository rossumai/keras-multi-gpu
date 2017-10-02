'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Lambda
from keras.layers.merge import concatenate
from keras.optimizers import RMSprop
from keras import backend as K
import os
import tensorflow as tf

from keras_tf_multigpu.kuza55 import make_parallel

# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
K.set_session(sess)

gpu_count = len([dev for dev in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if len(dev.strip()) > 0])

batch_size = 256 * gpu_count
layer_count = 5
layer_width = 1024
num_classes = 10
epochs = 5
ps_device = '/gpu:0'
# ps_device = '/cpu:0'

# gpu_count=1, batch_size=128, (width=1024) -> 11s / epoch

# measurement after 4th epoch, when it stabilizes
# PS: cpu:0
# gpu_count=1, batch_size=1024*N, (layers=10 x 2048), 39M params -> 5 s / epoch
# gpu_count=1, batch_size=1024*N, (layers=10 x 4096), 154M params -> 17 s / epoch
# gpu_count=2, batch_size=1024*N, (layers=10 x 4096), 154M params -> ~80 s / epoch (!)
# gpu_count=4, batch_size=1024*N, (layers=10 x 4096), 154M params -> ~40 s / epoch (!)
# big model (10x4096), small batch size (64*N) -> slow

# PS: gpu:0
# gpu_count=1, batch_size=256*N, (layers=5 x 1024), 5M params -> 2 s / epoch
# gpu_count=2, batch_size=256*N, (layers=5 x 1024), 5M params -> 7 s / epoch (!)
# gpu_count=2, batch_size=256 (2x128), (layers=5 x 1024), 5M params -> 13 s / epoch (!)
# gpu_count=1, batch_size=128, (layers=5 x 1024), 5M params -> 4 s / epoch

# 2017-08-22
# measurement after 4th epoch, when it stabilizes

# PS: gpu:0
# gpu_count=1, batch_size=256, (layers=5 x 1024), 5M params -> 2 s / epoch
# gpu_count=2, batch_size=256, (layers=5 x 1024), 5M params -> 14 s / epoch
# gpu_count=4, batch_size=256, (layers=5 x 1024), 5M params -> 41 s / epoch

# gpu_count=1, batch_size=32, (layers=5 x 1024), 5M params -> 16 s / epoch
# gpu_count=1, batch_size=64, (layers=5 x 1024), 5M params -> 8 s / epoch
# gpu_count=1, batch_size=128, (layers=5 x 1024), 5M params -> 4 s / epoch
# gpu_count=1, batch_size=256, (layers=5 x 1024), 5M params -> 2 s / epoch
# gpu_count=1, batch_size=512, (layers=5 x 1024), 5M params -> 1 s / epoch

# gpu_count=2, batch_size=32 * 2, (layers=5 x 1024), 5M params -> 53 s / epoch
# gpu_count=2, batch_size=64 * 2, (layers=5 x 1024), 5M params -> 27 s / epoch
# gpu_count=2, batch_size=128 * 2, (layers=5 x 1024), 5M params -> 13 s / epoch
# gpu_count=2, batch_size=256 * 2, (layers=5 x 1024), 5M params -> 7 s / epoch
# gpu_count=2, batch_size=512 * 2, (layers=5 x 1024), 5M params -> 5 s / epoch

# gpu_count=4, batch_size=64 * 4, (layers=5 x 1024), 5M params -> 41 s / epoch
# gpu_count=4, batch_size=256 * 4, (layers=5 x 1024), 5M params -> 10 s / epoch

# Gradients seem to be evenly distributed!

# PS: cpu:0

# gpu_count=1, batch_size=256, (layers=5 x 1024), 5M params -> 16 s / epoch
# gpu_count=2, batch_size=256 * 2, (layers=5 x 1024), 5M params -> 8 s / epoch

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

def basic_model():
    input = Input(shape=(784,))
    x = input
    for i in range(layer_count):
        x = Dense(layer_width, activation='relu')(x)
        x = Dropout(0.2)(x)
    output = Dense(10, activation='softmax')(x)

    model = Model(inputs=input, outputs=output)

    print('Single tower model:')
    model.summary()
    return model

tensorboard_dir = './tensorboard-logs/mnist_mlp_multi_ps_cpu/%d-gpu_%s' \
    % (gpu_count, os.environ.get('CUDA_VISIBLE_DEVICES', ''))

with tf.device(ps_device):
    serial_model = basic_model()
    print('Serial model:')
    serial_model.summary()

    model = make_parallel(serial_model, gpu_count, ps_device)
    print('Multi-GPU model:')
    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    summary_writer = tf.summary.FileWriter(tensorboard_dir, sess.graph)
    summary_writer.flush()

    tensorboard_cb = keras.callbacks.TensorBoard(log_dir=tensorboard_dir)
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test),
                        # callbacks=[tensorboard_cb])
                        callbacks=[])
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
