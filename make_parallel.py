# adapted from: https://github.com/kuza55/keras-extras/blob/master/utils/multi_gpu.py
#
## Example usage:
#
# # parameter server device
# ps_device = '/gpu:0'
#
# def basic_model():
#     #...
#
# with tf.device(ps_device):
#     model = make_parallel(basic_model(), gpu_count, ps_device)
#     model.compile(loss='categorical_crossentropy', optimizer='sgd')

from keras.layers.merge concatenate
from keras.layers import Lambda
from keras.models import Model
import tensorflow as tf

def make_parallel(model, gpu_count, ps_device=None):
    if gpu_count <= 1:
        return model

    if ps_device is None:
        ps_device = '/gpu:0'

    def get_slice(data, idx, parts):
        shape = tf.shape(data)
        size = tf.concat([shape[:1] // parts, shape[1:]], axis=0)
        stride = tf.concat([shape[:1] // parts, shape[1:] * 0], axis=0)
        start = stride * idx
        return tf.slice(data, start, size)

    outputs_all = []
    for i in range(len(model.outputs)):
        outputs_all.append([])

    # Place a copy of the model on each GPU, each getting a slice of the batch
    for i in range(gpu_count):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i) as scope:

                inputs = []
                # Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_n = Lambda(get_slice, output_shape=input_shape,
                        arguments={'idx': i, 'parts': gpu_count})(x)
                    inputs.append(slice_n)

                outputs = model(inputs)

                if not isinstance(outputs, list):
                    outputs = [outputs]

                # Save all the outputs for merging back together later
                for l in range(len(outputs)):
                    outputs_all[l].append(outputs[l])

    # merge outputs on parameter server
    with tf.device(ps_device):
        merged = []
        for outputs in outputs_all:
            merged.append(concatenate(outputs, axis=0))

        return Model(inputs=model.inputs, outputs=merged)
