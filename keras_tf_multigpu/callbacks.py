from __future__ import print_function

import time

import numpy as np
from keras.callbacks import Callback
import keras.backend as K
import tensorflow as tf

# NOTE: So far we observed asynchronous feeding for StagingAreaCallback.
# There's a simpligied implementation StagingAreaCallbackFeedDict which uses
# feed_dict instead of intermediate tf.Variable, but it's still synchronous.

class StagingAreaCallback(Callback):
    """
    It allows to prefetch input batches to GPU using TensorFlow StagingArea,
    making a simple asynchronous pipeline.

    The classic mechanism of copying input data to GPU in Keras with TensorFlow
    is `feed_dict`: a numpy array is synchronously copied from Python to TF memory
    and then using a host-to-device memcpy to GPU memory. The computation,
    however has to wait, which is wasteful.

    This class makes the HtoD memcpy asynchronous using a GPU-resident queue
    of size two (implemented by StaginArea). The mechanism is as follows:

    - at the beginning of an epoch one batch is `put()` into the queue
    - during each training step another is is `put()` into the queue and in
      parallel the batch already present at the GPU is `get()` from the queue
      at provide as tesnor input to the Keras model (this runs within a single
      `tf.Session.run()`)

    The input numpy arrays (features and targets) are provided via this
    callback and sliced into batches inside it. The last batch might be of
    smaller size without any problem (the StagingArea supports variable-sized
    batches and allows to enforce constant data sample shape). In the last
    batch zero-length slice is still put into the queue to keep the get+put
    operation uniform across all batches.

    Since it's hard to modify Keras to add more data to `feed_dict`, the data
    from numpy is fed into StagingArea in another `tf.Session.run()` before each
    training step via an intermediate `tf.Variable` and `feed_dict`. It is still
    synchronous. A better, though more complicated way would be to use TF queues
    (depracated) or Dataset API.

    In order to provide extra put() operation to `fetches`, we depend on a fork
    of Keras (https://github.com/bzamecnik/keras/tree/tf-function-session-run-args).
    A pull request to upstream will be made soon.

    Example usage:

    ```
    staging_area_callback = StagingAreaCallback(x_train, y_train, batch_size)

    image = Input(tensor=staging_area_callback.input_tensor)
    x = Dense(512, activation='relu')(image)
    digit = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=image, outputs=digit)

    model.compile(optimizer='sgd', loss='categorical_crossentropy',
        target_tensors=[staging_area_callback.target_tensor],
        fetches=staging_area_callback.extra_ops)

    model.fit(steps_per_epoch=steps_per_epoch, epochs=2,
        callbacks=[staging_area_callback])
    ```

    Full example: https://gist.github.com/bzamecnik/b520e2b1e199b193b715477929e39b22
    """
    def __init__(self, x, y, batch_size, prefetch_count=1):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.prefetch_count = prefetch_count

        features_shape = (None,) + x.shape[1:]
        labels_shape = (None,) + y.shape[1:]

        with tf.device('/cpu:0'):
            # for feeding inputs to the the StagingArea
            # Let's try to decouple feeding data to StagingArea.put()
            # from the training batch session.run()
            # https://www.tensorflow.org/api_guides/python/reading_data#Preloaded_data
            self.features_batch_next_value = tf.placeholder(dtype=x.dtype, shape=features_shape)
            # - prevent the variable to be used as a model parameter: trainable=False, collections=[]
            # - allow dynamic variable shape (for the last batch): validate_shape=False
            features_batch_next = tf.Variable(self.features_batch_next_value, trainable=False, collections=[], validate_shape=False)
            self.labels_batch_next_value = tf.placeholder(dtype=y.dtype, shape=labels_shape)
            labels_batch_next = tf.Variable(self.labels_batch_next_value, trainable=False, collections=[], validate_shape=False)
        self.assign_next_batch = tf.group(features_batch_next.initializer, labels_batch_next.initializer)

        # will be used for prefetching to GPU
        area = tf.contrib.staging.StagingArea(
            dtypes=[x.dtype, y.dtype],
            shapes=[features_shape, labels_shape])

        self.area_put = area.put([features_batch_next.value(), labels_batch_next.value()])
        area_get_features, area_get_labels = area.get()
        self.area_size = area.size()
        self.area_clear = area.clear()

        self.input_tensor = area_get_features
        self.target_tensor = area_get_labels
        self.extra_ops = [self.area_put]

    def set_params(self, params):
        super().set_params(params)
        self.steps_per_epoch = self.params['steps']

    def _slice_batch(self, i):
        start = i * self.batch_size
        end = start + self.batch_size
        return (self.x[start:end], self.y[start:end])

    def _assign_batch(self, session, data):
        x_batch, y_batch = data
        session.run(self.assign_next_batch, feed_dict={
            self.features_batch_next_value: x_batch,
            self.labels_batch_next_value: y_batch})

    def on_epoch_begin(self, epoch, logs=None):
        sess = K.get_session()
        for i in range(self.prefetch_count):
            self._assign_batch(sess, self._slice_batch(i))
            sess.run(self.area_put)

    def on_batch_begin(self, batch, logs=None):
        sess = K.get_session()
        # Slice for `prefetch_count` last batches is empty.
        # It serves as a dummy value which is put into StagingArea
        # but never read.
        data = self._slice_batch(batch + self.prefetch_count)
        self._assign_batch(sess, data)

    def on_epoch_end(self, epoch, logs=None):
        sess = K.get_session()
        sess.run(self.area_clear)

class StagingAreaCallbackFeedDict(Callback):
    """
    It allows to prefetch input batches to GPU using TensorFlow StagingArea,
    making a simple asynchronous pipeline.

    The classic mechanism of copying input data to GPU in Keras with TensorFlow
    is `feed_dict`: a numpy array is synchronously copied from Python to TF memory
    and then using a host-to-device memcpy to GPU memory. The computation,
    however has to wait, which is wasteful.

    This class makes the HtoD memcpy asynchronous using a GPU-resident queue
    of size two (implemented by StaginArea). The mechanism is as follows:

    - at the beginning of an epoch one batch is `put()` into the queue
    - during each training step another is is `put()` into the queue and in
      parallel the batch already present at the GPU is `get()` from the queue
      at provide as tesnor input to the Keras model (this runs within a single
      `tf.Session.run()`)

    The input numpy arrays (features and targets) are provided via this
    callback and sliced into batches inside it. The last batch might be of
    smaller size without any problem (the StagingArea supports variable-sized
    batches and allows to enforce constant data sample shape). In the last
    batch zero-length slice is still put into the queue to keep the get+put
    operation uniform across all batches.

    We feed input data to StagingArea via `feed_dict` as an additional input
    besides Keras inputs. Note that the `feed_dict` dictionary is passed as a
    reference and its values are updated inside the callback. It is still
    synchronous. A better, though more complicated way would be to use TF queues
    (depracated) or Dataset API.

    It seems to help on GPUs with low host-device bandwidth, such as desktop
    machines with many GPUs sharing a limited number of PCIe channels.

    In order to provide extra put() operation to `fetches`, we depend on a fork
    of Keras (https://github.com/bzamecnik/keras/tree/tf-function-session-run-args).
    A pull request to upstream will be made soon.

    Example usage:

    ```
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
        callbacks=[staging_area_callback])
    ```

    Full example: https://gist.github.com/bzamecnik/b520e2b1e199b193b715477929e39b22
    """
    def __init__(self, x, y, batch_size, prefetch_count=1):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.prefetch_count = prefetch_count

        features_shape = (None,) + x.shape[1:]
        labels_shape = (None,) + y.shape[1:]

        # inputs for feeding inputs to the the StagingArea
        self.features_batch_next = tf.placeholder(dtype=x.dtype, shape=features_shape)
        self.labels_batch_next = tf.placeholder(dtype=y.dtype, shape=labels_shape)
        # We'll assign self.features_batch_next, self.labels_batch_next before
        # each StagingArea.put() - feed_dict is passed by reference and updated
        # from outside.
        self.feed_dict = {}

        # will be used for prefetching to GPU
        area = tf.contrib.staging.StagingArea(
            dtypes=[x.dtype, y.dtype],
            shapes=[features_shape, labels_shape])

        self.area_put = area.put([self.features_batch_next, self.labels_batch_next])
        area_get_features, area_get_labels = area.get()
        self.area_size = area.size()
        self.area_clear = area.clear()

        self.input_tensor = area_get_features
        self.target_tensor = area_get_labels
        self.extra_ops = [self.area_put]

    def set_params(self, params):
        super().set_params(params)
        self.steps_per_epoch = self.params['steps']

    def _slice_batch(self, i):
        start = i * self.batch_size
        end = start + self.batch_size
        return (self.x[start:end], self.y[start:end])

    def _update_feed_dict(self, data):
        x_batch, y_batch = data
        self.feed_dict[self.features_batch_next] = x_batch
        self.feed_dict[self.labels_batch_next] = y_batch

    def on_epoch_begin(self, epoch, logs=None):
        sess = K.get_session()
        # initially fill the StagingArea
        for i in range(self.prefetch_count):
            self._update_feed_dict(self._slice_batch(i))
            sess.run(feed_dict=self.feed_dict, fetches=[self.area_put])

    def on_batch_begin(self, batch, logs=None):
        sess = K.get_session()
        # Slice for `prefetch_count` last batches is empty.
        # It serves as a dummy value which is put into StagingArea
        # but never read.
        self._update_feed_dict(self._slice_batch(batch + self.prefetch_count))

    def on_epoch_end(self, epoch, logs=None):
        sess = K.get_session()
        sess.run(self.area_clear)

class BatchTiming(Callback):
    """
    It measure robust stats for timing of batches and epochs.
    Useful for measuring the training process.

    For each epoch it prints median batch time and total epoch time.
    After training it prints overall median batch time and median epoch time.

    Usage: model.fit(X_train, Y_train, callbacks=[BatchTiming()])

    All times are in seconds.

    More info: https://keras.io/callbacks/
    """
    def on_train_begin(self, logs={}):
        self.all_batch_times = []
        self.all_epoch_times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_batch_times = []

    def on_batch_begin(self, batch, logs={}):
        self.start_time = time.time()

    def on_batch_end(self, batch, logs={}):
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        self.epoch_batch_times.append(elapsed_time)
        self.all_batch_times.append(elapsed_time)

    def on_epoch_end(self, epoch, logs={}):
        epoch_time = np.sum(self.epoch_batch_times)
        self.all_epoch_times.append(epoch_time)
        median_batch_time = np.median(self.epoch_batch_times)
        print('Epoch timing - batch (median): %0.5f, epoch: %0.5f (sec)' % \
            (median_batch_time, epoch_time))

    def on_train_end(self, logs={}):
        median_batch_time = np.median(self.all_batch_times)
        median_epoch_time = np.median(self.all_epoch_times)
        print('Overall - batch (median): %0.5f, epoch (median): %0.5f (sec)' % \
            (median_batch_time, median_epoch_time))

class SamplesPerSec(Callback):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def on_train_begin(self, logs={}):
        self.all_samples_per_sec = []

    def on_batch_begin(self, batch, logs={}):
        self.start_time = time.time()
        # self.batch_size = logs['size']

    def on_batch_end(self, batch, logs={}):
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        samples_per_sec = self.batch_size / elapsed_time
        self.all_samples_per_sec.append(samples_per_sec)

    def on_epoch_end(self, epoch, logs={}):
        self.print_results()

    def print_results(self):
        print('Samples/sec: %0.2f' % np.median(self.all_samples_per_sec))

"""
Enables CUDA profiling (for usage in nvprof) just for a few batches.

The reasons are:

- profiling outputs are big (easily 100s MB - GBs) and repeating
- without a proper stop the outputs sometimes fail to save

Since initially the TensorFlow runtime may take time to optimize the graph we
skip a few epochs and then enable profiling for a few batches within the next
epoch.

It requires the `cudaprofile` package.
"""
class CudaProfile(Callback):

    def __init__(self, warmup_epochs=0, batches_to_profile=None):
        self.warmup_epochs = warmup_epochs
        self.batches_to_profile = batches_to_profile
        self.enabled = False

    def set_params(self, params):
        self.params = params

    def on_epoch_begin(self, epoch, logs={}):
        import cudaprofile
        if epoch == self.warmup_epochs:
            cudaprofile.start()
            self.enabled = True

    def on_batch_end(self, batch, logs={}):
        import cudaprofile
        if self.enabled and batch >= batches_to_profile:
            cudaprofile.stop()
