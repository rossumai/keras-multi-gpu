import keras.backend as K
from keras.layers import Lambda
from keras.layers.merge import concatenate
from keras.models import Model
import keras.optimizers
from keras.optimizers import clip_norm, Optimizer
import tensorflow as tf

# this should be fairly ready
class DataParallelOptimizer(Optimizer):
    """
    Wrapper class for data-parallel optimization. Multiple model replicas
    (towers) with shared weights operate on different batch slices and compute
    gradients in parallel on multiple GPUs. Gradients are then averaged on
    parameter sever (CPU or one of GPUs) and weights updated.

    It accepts a list of losses (living on separate devices) instead of a
    single loss, computes gradients (collocated with losses) for each loss,
    averages then on the PS device and provides weight update operations.

    Usage:
    from keras.optimizers import Adam
    model.compile(..., optimizer=DataParallelOptimizer(Adam()))
    """

    def __init__(self, optimizer):
        self.optimizer = keras.optimizers.get(optimizer)

    def get_gradients(self, losses, params):
        # NOTE: argument "losses" (list) instead of a single "loss"

        if isinstance(losses, list):
            # Gradients for each tower loss.
            # NOTE: K.gradients call tf.gradiens with
            # colocate_gradients_with_ops=True, thus each tf.gradient operation
            # should be collocated with it's respective loss. We assume losses
            # to be located at different devices.
            tower_grads = [K.gradients(loss, params) for loss in losses]
            # Average gradients.
            # This should be a synchronization point (for sync SGD) and this
            # operation will be located according to the scope where the main
            # Model was defined - should be the parameter server device.
            grads = K.mean(K.stack(tower_grads, 0))
        else:
            grads = K.gradients(losses, params)

        if hasattr(self, 'clipnorm') and self.clipnorm > 0:
            norm = K.sqrt(sum([K.sum(K.square(g)) for g in grads]))
            grads = [clip_norm(g, self.clipnorm, norm) for g in grads]
        if hasattr(self, 'clipvalue') and self.clipvalue > 0:
            grads = [K.clip(g, -self.clipvalue, self.clipvalue) for g in grads]
        return grads

    def get_updates(self, params, constraints, loss):
        return self.optimizer.get_updates(params, constraints, loss)

    @property
    def weights(self):
        self.optimizer.weights()

    def get_config(self):
        self.optimizer.get_config()

    def from_config(self, config):
        self.optimizer.from_config()

# so far just an incomplete sketch...
class DataParallelModel(Model):
    def __init__(self, inputs, outputs, basic_model, replicas, name=None):
        super(DataParallelModel, self).__init__(inputs, outputs, name)
        self.basic_model = basic_model
        self.replicas = replicas

    @classmethod
    def create(cls, basic_model, gpu_count=2):
        assert gpu_count >= 2, "At least 2 GPUs"

        def get_slice(data, idx, parts):
            shape = tf.shape(data)
            size = tf.concat([shape[:1] // parts, shape[1:]], axis=0)
            stride = tf.concat([shape[:1] // parts, shape[1:] * 0], axis=0)
            start = stride * idx
            return tf.slice(data, start, size)

        outputs_all = []
        replicas = []
        # place operations for replica on a separate device
        for gpu_id in range(gpu_count):
            with tf.device("gpu:%d" % gpu_id):
                with tf.name_scope('replica_%d' % gpu_id):

                    slices = []
                    # Slice each input into a piece for processing on this GPU
                    for x in basic_model.inputs:
                        input_shape = tuple(x.get_shape().as_list())[1:]
                        slice = Lambda(get_slice, output_shape=input_shape,
                            arguments={'idx': gpu_id, 'parts': gpu_count})(x)
                        slices.append(slice)

                    if gpu_id == 0:
                        for i in range(len(basic_model.outputs)):
                            outputs_all.append([])

                    outputs = basic_model(slices)
                    replica = Model(inputs=basic_model.inputs, outputs=outputs)
                    replicas.append(replica)

                    if not isinstance(outputs, list):
                        outputs = [outputs]

                    # Save all the outputs for merging back together later
                    for l in range(len(outputs)):
                        outputs_all[l].append(outputs[l])

        with tf.device("gpu:0"):
            merged = []
            for outputs in outputs_all:
                merged.append(concatenate(outputs, axis=0))
            return cls(inputs=basic_model.inputs, outputs=merged,
                basic_model=basic_model, replicas=replicas)

    def compile(self, optimizer, loss, metrics=None, loss_weights=None,
                sample_weight_mode=None, **kwargs):
        """
        optimizer - identifier or instance of an optimizer
        loss - identifier or instance of a loss function
        """

        replica_total_losses = []
        # place the loss and gradient operations for replica on a separate device
        for gpu_id, replica in enumerate(self.replicas):
            with tf.device("gpu:%d" % gpu_id):
                with tf.name_scope('replica_%d' % gpu_id):
                    replica.compile(optimizer, loss, metrics, loss_weights)
                    replica_total_losses.append(replica.total_loss)

        super(DataParallelModel, self).compile(
            DataParallelOptimizer(optimizer), loss, metrics, loss_weights)
        # separate losses whose gradient can be computed in parallel
        self.replica_total_losses = replica_total_losses
        # redefine total_loss with the average of replica losses
        self.total_loss = K.mean(K.stack(replica_total_losses, 0))

    def _make_train_function(self):
        if not hasattr(self, 'train_function'):
            raise RuntimeError('You must compile your model before using it.')
        if self.train_function is None:
            inputs = self._feed_inputs + self._feed_targets + self._feed_sample_weights
            if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
                inputs += [K.learning_phase()]

            assert isinstance(self.optimizer, DataParallelOptimizer)

            training_updates = self.optimizer.get_updates(
                self._collected_trainable_weights,
                self.constraints,
                self.replica_total_losses)
            updates = self.updates + training_updates
            # Gets loss and metrics. Updates weights at each call.
            self.train_function = K.function(inputs,
                                             [self.total_loss] + self.metrics_tensors,
                                             updates=updates,
                                             name='train_function',
                                             **self._function_kwargs)

# TODO: in ModelCheckpointer save the basic_model
