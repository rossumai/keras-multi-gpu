# Towards efficient multi-GPU training in Keras over TensorFlow

The big message:

Multi-GPU data-parallel training in Keras over TensorFlow is with some hacks possible at the moment, but not too efficient. So we provide an in-depth analysis of the problem and existing solutions, and suggest what to do to make it more effcient.

## Introduction

### Why?

In [Rossum](https://rossum.ai) we'd like to reduce training of our image-processing models from around a day to several hours. We use [Keras](https://keras.io) over [TensorFlow](https://www.tensorflow.org). At hand we have a custom-built machine with 7 GPUs (GTX 1070 and 1080) and common cloud instances with 2-4 GPUs (eg. Tesla M60). Can we utilize them to speed up the training?

It's not uncommon that bigger models take many hours to train. We already parallelize the computation by using many-core GPUs. Next step is using multiple GPUs, then possibly multiple machines. Although we introduce more complexity, we expect decrease in training time. Ideally the training should scale inversely and the throughput of data samples per second linearly with respect to the number of GPUs. Of course we also expect some overhead of operations that cannot be parallelized.

[TenslowFlow shows](https://www.tensorflow.org/performance/benchmarks) achieving almost linear speed-up in their benchmarks of high-performance implementations of non-trivial models on up to cloud 8 GPUs.

Can we do similarly in our models using Keras over TensorFlow? So far we have a lot of models written in Keras running over TensorFlow and we chose these technologies for long term. Although rewriting to pure TensorFlow might be possible, we'd lose a lot of comfort of high-level API of Keras and it's benefits like callbacks, sensible defaults, etc.

### Scope

We limit the scope of this effort to training on single machine with multiple GPUs (not distributed), using data parallelism and we want the resulting implementation in Keras over TensorFlow.

### Organization of the article

Since the whole topic is a bit complex the main story is outlined in this article and details have been extracted into several separate articles. First we review existing algorithms a techniques, then existing implementations of multi-GPU training in common deep learning frameworks. We need to consider hardware since the performance heavily depends on it. To get intuition on what techniques are working well and how we perform and evaluate various measurements of existing implementations. For that we figure out what architectures and datasets are suitable for benchmarking. Then we finally review and measure existing approaches of multi-GPU training specifically in Keras + TensorFlow and indicate their problems. Finally we suggest which techniques might help.

- [Algorithms and techniques](algorithms-and-techniques.md)
- [Other implementations](other-implementations.md)
- [Hardware](hardware.md)
- [Baseline measurements](measurements.md)
- Keras + TensorFlow - review and measurements
- Suggestions how to improve
- Conclusion

## Keras over TensorFlow

Unlike many deep learning frameworks, Keras officially doesn't support multi-GPU training. There seem to be two reasons: Keras was designed for CPU and single-GPU training and and it's still highly specific to the underlying backend.

### Basic example from @fcholet

When François Chollet (@fchollet), the author of Keras author introduced the TensorFlow backend in [Keras as a simplified interface to TensorFlow: tutorial - Multi-GPU and distributed training](https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html#multi-gpu-and-distributed-training) (in April 2016) he also showed examples of model- and data-parallelism in Keras by breaking the abstraction and using some TensorFlow-specific stuff. It uses Keras just to build the model but uses TensorFlow to perform the training (thus leaving some Keras fancy stuff apart).

For data-parallelism it goes with parameter server on CPU with implicit copy and synchronous data feeding. In particular it makes one instance of the models located at the CPU. Then it creates two replicas places on GPUs that place only operations on GPUs but share the variables with CPU-located model. In the end it merges (averages) predictions on both replicas (again on the CPU) and evaluates that with `session.run()` where data are provided via `feed_dict`.

In this simple illustration example the global mini-batch is not actually scattered to GPUs but constant. Also it doesn't deal with computing loss and its gradients at all. So it can be though as just a sketch.

### Issues on Keras git repo

Several issues appeared in Keras repo addressing the need for multi-GPU training:

- [Does Keras support using multiple GPUs? #2436](https://github.com/fchollet/keras/issues/2436) (April 2016)
- [[WIP] Correct data-parallel SGD implementation in Keras #7515](https://github.com/fchollet/keras/issues/7515) (August 2017)
- [Data parallelism with multiple GPUs with fit_generator #7698](https://github.com/fchollet/keras/issues/7698) (August 2017)

Within the issue `#2436` thread evolved some code developing more practically on the example from @fchollet - function called `make_parallel()` used to replicate a model to multiple GPUs and scatter the input mini-batch.

### @kuza55 make_parallel()

In October 2016 @kuza55 published a blog article [Transparent Multi-GPU Training on TensorFlow with Keras](https://medium.com/@kuza55/transparent-multi-gpu-training-on-tensorflow-with-keras-8b0016fd9012) showing `make_parallel()` in practice along with some code [multi_gpu.py](https://github.com/kuza55/keras-extras/blob/master/utils/multi_gpu.py) and pull request [3620](https://github.com/fchollet/keras/pull/3620). The problem was after compiling the model gradients single loss function ended up at one device. By setting TensorFlow option `colocate_gradients_with_ops=True` gradients should be evenly distributed across the replica devices.

Even though the code at least runs it has some drawbacks:

- instead of really scattering the sub-batches the whole mini-batch is copied to all GPUs and then sliced
  - this wastes precious bandwidth a lot
- it fails on datasets that are not evenly divisible by the sub-batch size
- it still only concatenates the predictions and does not truly perform gradient averaging
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

... TODO ...
