# Keras over TensorFlow

Unlike many deep learning frameworks, Keras officially doesn't support multi-GPU training. There seem to be two reasons: Keras was designed for CPU and single-GPU training and and it's still highly specific to the underlying backend.

### Basic example from @fcholet

When François Chollet (@fchollet), the author of Keras author introduced the TensorFlow backend in [Keras as a simplified interface to TensorFlow: tutorial - Multi-GPU and distributed training](https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html#multi-gpu-and-distributed-training) (in April 2016) he also showed examples of model- and data-parallelism in Keras by breaking the abstraction and using some TensorFlow-specific stuff. It uses Keras just to build the model but uses TensorFlow to perform the training (thus leaving some Keras fancy stuff apart).

For data-parallelism it goes with parameter server on CPU with implicit copy and synchronous data feeding. In particular it makes one instance of the models located at the CPU. Then it creates two replicas places on GPUs that place only operations on GPUs but share the variables with CPU-located model. In the end it merges (averages) predictions on both replicas (again on the CPU) and evaluates that with `session.run()` where data are provided via `feed_dict`.

In this simple illustration example the global mini-batch is not actually scattered to GPUs but constant. Also it doesn't deal with computing loss and its gradients at all. So it can be though as just a sketch.

### Issues on Keras git repo

Several issues appeared in Keras repo addressing the need for multi-GPU training:

- [Does Keras support using multiple GPUs? #2436](https://github.com/fchollet/keras/issues/2436) (April 2016)
- []Multi-GPU support for Tensorflow Backend(https://github.com/fchollet/keras/issues/3331) (Jul 2016)
- [[WIP] Correct data-parallel SGD implementation in Keras #7515](https://github.com/fchollet/keras/issues/7515) (August 2017)
- [Data parallelism with multiple GPUs with fit_generator #7698](https://github.com/fchollet/keras/issues/7698) (August 2017)

Within the issue `#2436` thread evolved some code developing more practically on the example from @fchollet - function called `make_parallel()` used to replicate a model to multiple GPUs and scatter the input mini-batch. ([@jonilaserson](https://github.com/fchollet/keras/issues/2436#issuecomment-256679606), @anewlearner, @kuza55, @avolkov1).

### @kuza55 make_parallel()

In October 2016 @kuza55 published a blog article [Transparent Multi-GPU Training on TensorFlow with Keras](https://medium.com/@kuza55/transparent-multi-gpu-training-on-tensorflow-with-keras-8b0016fd9012) showing `make_parallel()` in practice along with some code [multi_gpu.py](https://github.com/kuza55/keras-extras/blob/master/utils/multi_gpu.py) and pull request [3620](https://github.com/fchollet/keras/pull/3620). The problem was after compiling the model gradients single loss function ended up at one device. By setting TensorFlow option `colocate_gradients_with_ops=True` gradients should be evenly distributed across the replica devices.

![Gradients are indeed distributed](https://user-images.githubusercontent.com/446124/29559422-5d1283c2-872f-11e7-8958-76ae068818c7.png)

Even though the code at least runs it has some drawbacks:

- instead of really scattering the sub-batches the whole mini-batch is copied to all GPUs and then sliced
  - this wastes precious bandwidth a lot
  - @avolkov1 solves this by [slicing on the cpu device](https://github.com/fchollet/keras/issues/2436#issuecomment-300913250).
- it fails on datasets that are not evenly divisible by the sub-batch size
- it still only concatenates the predictions and does not truly perform gradient averaging
  - does this hurt in any way?
- parameter server is defined rather implicitly according to the passed model + hardcoded to CPU, while merged outputs are hardcoded to CPU

### Our attempts

In July 2017 I started fillding with multi-GPU training in Keras. After identifying the problems above I set up an issue - [Correct data-parallel SGD implementation in Keras #7515](https://github.com/fchollet/keras/issues/7515) to explicitly address them.

At first by logging device placement and examining the model in TensorBoard it seemed that gradients in kuza55 were not really distributed, so I tried to write my own data-parallel wrapper class [DataParallelModel](https://github.com/rossumai/keras-multi-gpu/blob/master/keras_tf_multigpu/bzamecnik/data_parallel_model.py) to do exactly that. I had to change Optimizer to perform gradient averaging and `Model.compile()` to compute separate losses and use a wrapped optimizer.

To my surprise in Keras 2.0.7 with "Improve TensorBoard UX with better grouping of ops into name scopes." the gradients appreaded to be [really evenly distributed](https://github.com/fchollet/keras/issues/7515#issuecomment-323980530), just as other users reported. Thus doing explicit gradient averaging seems unnecessary in Keras and DataParallelModel has been put into an icebox.

In the meanwhile Keras+MXNet and Keras+CNTK appeared which showed almost ideal scaling. Also `avolkov1` repository appeared which claimed to support TF queues and NCCL and users were reporting good scaling.

From examining kuza55 closely via `nvprof` profiler it could be seen that the computation is waiting for data most of the time. Thus either the machine has low bandwidth or we need asyncrhronous data feeding or both.

### Synchronous data feeding

Unfortunately this is exactly how [Keras over TensorFlow feeds the data](https://github.com/fchollet/keras/blob/2.0.8/keras/backend/tensorflow_backend.py#L2272) in it's default implementation: it passes data via `feed_dict` argument of [`session.run()`](https://www.tensorflow.org/api_docs/python/tf/Session#run).

```
# keras.backend.tensorflow.Function.__call__():
updated = session.run(self.outputs + [self.updates_op],
                              feed_dict=feed_dict,
                              **self.session_kwargs)
```

This is exactly what's not recommended for in the TF high-performance guide. Instead we should use queues to feed the data asynchronously so that the computation kernels do not have to wait for the next batch.

It seems that in order to improve performance for single-GPU and mainly multi-GPU training we need to use asynchronous data feeding.

### @avolkov1 make_parallel()

In the meanwhile appeared another repository [avolkov1/keras_experiments](https://github.com/avolkov1/keras_experiments) with improved `make_parallel()` function which claims to support even NCCL, some examples using TF queue and even distributed training.

Some observations:

- uses TF `StagingArea` to load data
    - however the implementation is isn't implemented properly so far (issue #2)
    - monkey-patches keras backend Function to pass enqueue operations to `session.run()`
- uses NCCL (via TensorFlow) to broadcast parameters from one tower to others
- it uses method `load()`/`save()` from the serial model so that we store only one set of weights

### @fcholet's variant of @kuza55 script in keras.utils.training_utils.multi_gpu_model()

2017-10-13:

- https://twitter.com/fchollet/status/918205049225936896
- https://github.com/fchollet/keras/blob/3dd3e8331677e68e7dec6ed4a1cbf16b7ef19f7f/keras/utils/training_utils.py#L56-L75

- nicely refactored version of kuza55
- devices are explored via TF device_lib, instead of CUDA_VISIBLE_DEVICES
- (-) doesn't solve saving/loading weights
- (-) slicing is done on GPU, should be on CPU


### Horovod

2017010-05 - @alsrgv: https://github.com/uber/horovod/blob/master/examples/keras_mnist.py

### TensorPack
