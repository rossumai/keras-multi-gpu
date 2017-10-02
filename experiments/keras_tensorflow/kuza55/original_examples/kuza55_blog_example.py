# Multi-GPU Keras training.
# Captured from https://github.com/kuza55/keras-extras and
# https://medium.com/@kuza55/transparent-multi-gpu-training-on-tensorflow-with-keras-8b0016fd9012.
#
# Adapted for easier measurement: arg parser, etc.

import argparse
import numpy as np
import keras
from keras.layers import merge, Dense
from keras.layers.core import Lambda
from keras.models import Model, Sequential
import tensorflow as tf

# source: https://github.com/kuza55/keras-extras/blob/master/utils/multi_gpu.py
def make_parallel(model, gpu_count, ps_device):
    def get_slice(data, idx, parts):
        shape = tf.shape(data)
        size = tf.concat([ shape[:1] // parts, shape[1:] ],axis=0)
        stride = tf.concat([ shape[:1] // parts, shape[1:]*0 ],axis=0)
        start = stride * idx
        return tf.slice(data, start, size)

    outputs_all = []
    for i in range(len(model.outputs)):
        outputs_all.append([])

    #Place a copy of the model on each GPU, each getting a slice of the batch
    for i in range(gpu_count):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i) as scope:

                inputs = []
                #Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_n = Lambda(get_slice, output_shape=input_shape, arguments={'idx':i,'parts':gpu_count})(x)
                    inputs.append(slice_n)

                outputs = model(inputs)

                if not isinstance(outputs, list):
                    outputs = [outputs]

                #Save all the outputs for merging back together later
                for l in range(len(outputs)):
                    outputs_all[l].append(outputs[l])

    # merge outputs on CPU
    with tf.device(ps_device):
        merged = []
        for outputs in outputs_all:
            merged.append(merge(outputs, mode='concat', concat_axis=0))

        return Model(input=model.inputs, output=merged)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='kuza55 make_parallel test.')
    parser.add_argument('-g', '--gpus', default=1, type=int, help='Number of GPUs to use')
    parser.add_argument('-p', '--parameter-server', default='cpu', help='Parameter server device (cpu, gpu)')
    parser.add_argument('-e', '--epochs', default=5, type=int, help='Number of epochs')
    args = parser.parse_args()
    ps_device = '/%s:0' % args.parameter_server

    # source: https://medium.com/@kuza55/transparent-multi-gpu-training-on-tensorflow-with-keras-8b0016fd9012
    with tf.device(ps_device):
        model = Sequential()
        model.add(Dense(4000, input_dim=8000, activation='tanh'))
        model.add(Dense(2000, input_dim=8000, activation='relu'))
        model.add(Dense(500, activation='relu'))
        model.add(Dense(300, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        print (model.summary())
        if args.gpus > 1:
            model = make_parallel(model, args.gpus, ps_device)
        optimizer = keras.optimizers.Adam(lr=0.0001)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    x = np.random.rand(131072, 8000)
    y = np.random.randint(0, 2, (131072, 1))

    model.fit(x, y, batch_size=2048*args.gpus, epochs=args.epochs)
