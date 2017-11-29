# Example of using StagingAreaCallback for GPU prefetch on a simple convnet
# model on CIFAR10 dataset.
#
# https://gist.github.com/bzamecnik/b9dbd50cdc195d54513cd2f9dfb7e21b

import math

from keras.layers import Dense, Input, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.models import Model
from keras.utils import to_categorical
import numpy as np

from keras_tf_multigpu.callbacks import StagingAreaCallbackFeedDict, SamplesPerSec
from keras_tf_multigpu.examples.datasets import create_synth_cifar10

np.random.seed(42)

def make_convnet(input, num_classes):
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(input)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)

    return output

def make_plain_model(input_shape, num_classes):
    input = Input(shape=input_shape)
    model = Model(inputs=input, outputs=make_convnet(input, num_classes))
    model.compile(optimizer='sgd', loss='categorical_crossentropy')
    return model

def make_tensor_model(staging_area_callback, num_classes):
    input = Input(tensor=staging_area_callback.input_tensor)
    model = Model(inputs=input, outputs=make_convnet(input, num_classes))
    model.compile(optimizer='sgd', loss='categorical_crossentropy',
        target_tensors=[staging_area_callback.target_tensor],
        feed_dict=staging_area_callback.feed_dict,
        fetches=staging_area_callback.extra_ops)
    return model

num_classes = 10
dataset_size = 50000
batch_size = 2048
epochs = 5

x_train, y_train = create_synth_cifar10(dataset_size)
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
