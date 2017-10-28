# Example of using StagingAreaCallback for GPU prefetch on ResNet50 model on
# synthetic ImageNet dataset.

# https://gist.github.com/bzamecnik/12022c8dd50ec8eda0c2661830bbc8a4

import math

from keras.applications import ResNet50
from keras.datasets import mnist
from keras.layers import Dense, Input
from keras.models import Model
from keras.utils import to_categorical
import numpy as np

from keras_tf_multigpu.callbacks import StagingAreaCallback, SamplesPerSec
from keras_tf_multigpu.examples.datasets import create_synth_imagenet


def make_plain_model(num_classes):
    model = ResNet50(classes=num_classes, weights=None)
    model.compile(optimizer='sgd', loss='categorical_crossentropy')
    return model

def make_tensor_model(features_tensor, targets_tensor, extra_ops, num_classes):
    model = ResNet50(input_tensor=features_tensor, classes=num_classes, weights=None)
    model.compile(optimizer='sgd', loss='categorical_crossentropy',
        target_tensors=[targets_tensor], fetches=extra_ops)
    return model

num_classes = 1000
dataset_size = 1024
batch_size = 32
epochs = 5

x_train, y_train = create_synth_imagenet(224, dataset_size)
x_train = x_train.astype('float32')
y_train = y_train.astype('float32')

# last batch might be smaller
steps_per_epoch = int(math.ceil(len(x_train) / batch_size))

gauge = SamplesPerSec(batch_size)
staging_area_callback = StagingAreaCallback(x_train, y_train, batch_size)

print('training plain model:')
plain_model = make_plain_model(x_train.shape[1:], num_classes)
plain_model.fit(x_train, y_train, batch_size, epochs=epochs, callbacks=[gauge])

print('training pipelined model:')
pipelined_model = make_tensor_model(staging_area_callback, num_classes)
pipelined_model.fit(steps_per_epoch=steps_per_epoch, epochs=epochs,
    callbacks=[staging_area_callback, gauge])
