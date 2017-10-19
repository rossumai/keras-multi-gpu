import tensorflow as tf
from keras.applications import InceptionV3
from keras.utils import multi_gpu_model
import numpy as np

num_samples = 1000
height = 224
width = 224
num_classes = 1000

gpu_count = 2

# Instantiate the base model
# (here, we do it on CPU, which is optional).
with tf.device('/cpu:0' if gpu_count > 1 else '/gpu:0'):
    model = InceptionV3(weights=None,
                     input_shape=(height, width, 3),
                     classes=num_classes)

# Replicates the model on N GPUs.
# This assumes that your machine has N available GPUs.
if gpu_count > 1:
    parallel_model = multi_gpu_model(model, gpus=gpu_count)
else:
    parallel_model = model
parallel_model.compile(loss='categorical_crossentropy',
                       optimizer='rmsprop')

# Generate dummy data.
x = np.random.random((num_samples, height, width, 3))
y = np.random.random((num_samples, num_classes))

# This `fit` call will be distributed on N GPUs.
# Since the batch size is N*32, each GPU will process 32 samples.
parallel_model.fit(x, y, epochs=20, batch_size=32 * gpu_count)
