# Example of using StagingAreaCallback for GPU prefetch on a trivial model
# on MNIST dataset.
#
# https://gist.github.com/bzamecnik/b520e2b1e199b193b715477929e39b22

import math
from keras.datasets import mnist
from keras.layers import Dense, Input
from keras.models import Model
from keras.utils import to_categorical
import numpy as np

from keras_tf_multigpu.callbacks import StagingAreaCallback, SamplesPerSec

np.random.seed(42)

num_classes = 10

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784).astype('float32') / 255
y_train = to_categorical(y_train, num_classes).astype('float32')

batch_size = 64 # makes the last batch of size 32

# last batch might be smaller
steps_per_epoch = int(math.ceil(len(x_train) / batch_size))

gauge = SamplesPerSec(batch_size)
staging_area_callback = StagingAreaCallback(x_train, y_train, batch_size)

image = Input(tensor=staging_area_callback.input_tensor)
x = Dense(512, activation='relu')(image)
digit = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=image, outputs=digit)

model.compile(optimizer='sgd', loss='categorical_crossentropy',
    target_tensors=[staging_area_callback.target_tensor],
    feed_dict=staging_area_callback.feed_dict,
    fetches=staging_area_callback.extra_ops)

model.fit(steps_per_epoch=steps_per_epoch, epochs=2,
    callbacks=[staging_area_callback, gauge])
