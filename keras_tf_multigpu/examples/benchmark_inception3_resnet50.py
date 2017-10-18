# InceptionV3:
# Total params: 23,851,784
# Trainable params: 23,817,352
# Non-trainable params: 34,432
# ~22M params in conv layers, ~2M params in last dense layer
#
# on Azure 1x M60:
# 1 GPU:             ~32 samples/sec, epoch time=35s
#
# method = kuza55:
# 2 GPU (PS at CPU): ~55 samples/sec, epoch time=19s (1.72x speed-up)
# 2 GPU (PS at GPU): ~53 samples/sec, epoch time=20s (1.66x speed-up)
#
# method = avolkov1:
# 2 GPU (PS at CPU): ~54 samples/sec, epoch time=19s (1.72x speed-up)
# 2 GPU (PS at GPU): ~53 samples/sec, epoch time=20s (1.66x speed-up)
#
# Result: there's some significant speed-up but not perfect.
#
# Comparison with TensorFlow benchmark:
# 1 GPU: 47.49 samples/sec
# 2 GPU (PS at CPU): 95.43 samples/sec

from __future__ import print_function
import argparse
from keras.applications import InceptionV3, ResNet50
import numpy as np
import tensorflow as tf

from keras_tf_multigpu.callbacks import SamplesPerSec
from keras_tf_multigpu.examples.datasets import create_synth_imagenet

# -- data-parallel model --

def train(create_model, X, y, batch_size, epochs, gpu_count, parameter_server, method):
    if gpu_count > 1:
        ps_device = '/gpu:0' if parameter_server == 'gpu' else '/cpu:0'

        with tf.device(ps_device):
            serial_model = create_model()

        if method == 'kuza55':
            from keras_tf_multigpu.kuza55 import make_parallel
            model = make_parallel(serial_model, gpu_count=gpu_count, ps_device=ps_device)
        elif method == 'avolkov1':
            from keras_tf_multigpu.avolkov1 import make_parallel, get_available_gpus
            gpus_list = get_available_gpus(gpu_count)
            model = make_parallel(serial_model, gdev_list=gpus_list, ps_device=ps_device)
    else:
        model = serial_model = create_model()

    print('Number of parameters:', serial_model.count_params())

    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    gauge = SamplesPerSec(batch_size)
    model.fit(X, y, batch_size=batch_size, epochs=epochs, callbacks=[gauge])
    gauge.print_results()

def parse_args():
    parser = argparse.ArgumentParser(description='Inception v3 / ResNet50 data-parallel in Keras')
    parser.add_argument('-a', '--arch', required=True, help='Architecture (inception3, resnet50)')
    parser.add_argument('-g', '--gpus', default=1, type=int, help='Number of GPUs to use')
    parser.add_argument('-p', '--parameter-server', default='gpu', help='Parameter server device (cpu, gpu)')
    parser.add_argument('-m', '--method', default='kuza55', help='Method of parallelization (kuza55, avolkov1)')
    parser.add_argument('-e', '--epochs', default=5, type=int, help='Number of epochs')
    # Note: batch_size == 64 with inception3 is too much for 8GB GPU RAM
    parser.add_argument('-b', '--batch-size', default=32, type=int, help='Batch size')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    print('architecture:', args.arch)
    print('number of GPUs:', args.gpus)
    print('parameter server:', args.parameter_server)
    print('parallelization method:', args.method)
    print('epochs:', args.epochs)
    print('batch_size:', args.batch_size)

    dataset_size = 1024
    if args.arch == 'inception3':
        create_model = InceptionV3
        X, y = create_synth_imagenet(299, dataset_size)

    elif args.arch == 'resnet50':
        create_model = ResNet50
        X, y = create_synth_imagenet(224, dataset_size)

    train(create_model, X, y, args.batch_size, args.epochs, args.gpus, args.parameter_server, args.method)
