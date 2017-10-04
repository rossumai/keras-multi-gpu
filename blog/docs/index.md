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

Synchronous data feeding:

Unfortunately this is exactly how [Keras over TensorFlow feeds the data](https://github.com/fchollet/keras/blob/2.0.8/keras/backend/tensorflow_backend.py#L2272) in it's default implementation: it passes data via `feed_dict` argument of [`session.run()`](https://www.tensorflow.org/api_docs/python/tf/Session#run).

```
# keras.backend.tensorflow.Function.__call__():
updated = session.run(self.outputs + [self.updates_op],
                              feed_dict=feed_dict,
                              **self.session_kwargs)
```
