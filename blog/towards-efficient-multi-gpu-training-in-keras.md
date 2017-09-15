# [WIP] Towards efficient multi-GPU training in Keras over TensorFlow

In this article I'd like to summarize the current state of attempts for data-parallel multi-GPU training in Keras over TensorFlow and a roadmap of steps to make it efficient and usable in practice.

We will cover two problems: efficiently training on a single GPU and then extending training to multiple GPUs on the same machine.

## Introduction

### Why?

It's not uncommon that bigger models take many hours to train. We already parallelize the computation by using many-core GPUs. Next step is using multiple GPUs. Although we introduce more complexity, we expect decrease in training time. Ideally the training should scale inversely and the throughput of data samples per second linearly with respect to the number of GPUs. Of course we also expect some overhead of operations that cannot be parallelized.

### Efficient GPU training with Keras + TensorFlow

https://www.tensorflow.org/performance/performance_models

### Ways of parallelism in deep learning

There are two principial ways how to parallelize traning of deep learning models:

- *data parallelism* - parts of data mini-batch are distributed over GPUs and trained on the same model
- *model parallelism* - parts of the model are distributed over GPUs and trained on the same mini-batch of data

Nice overview is an a series of articles by Tim Dettmers: [Data Parallelism](http://timdettmers.com/2014/10/09/deep-learning-data-parallelism/), [Model Parallelism](http://timdettmers.com/2014/11/09/model-parallelism-deep-learning/). It's recommended to use data-parallelism for convolutional and recurrent networks with a lot of computation and not too much parameters, while model parallelism for networks with too many parameters that would not fit within a single GPU memory (eg. layers of LSTM in GNMT).

#### Limiting the scope

Since our motivation in @rossumai is to speed up training of some bigger convolutional image models in this article we will focus only on data-parallelism. Also we restrict the solution to Keras over TensorFlow backend. Also we will not consider training distributed to mulitple machines.

### Algorithms for data-parallel training

At hand we have several devices: CPU and N GPUs.

A basic extension of SGD to data-parallel training iterates the following steps:

- weights are stored on a *parameter server* (PS) device (either CPU or one of the GPUs) and replicated to each GPU
- each GPU gets a slice of the mini-batch (a *sub-batch*)
- the GPUs compute in parallel the loss and gradients of the loss with respect to each weight
- the gradients are averaged, end up at PS device
- weights updated using gradients according to an update rule (momentum, RMSProp, ADAM, etc.)
- the updated weigths are sent to each GPU again

In the case when we wait for all devices to finish before doing the update we call it *synchronous SGD*, otherwise if any device can update the parameters independently we call it *asynchronous SGD*. Since we assume all GPUs are the same (homogeneous) and take the same time of a batch we can simply use sync SGD.

A nice intro is written in a [TensorFlow tutorial on CNNs](https://www.tensorflow.org/tutorials/deep_cnn#training_a_model_using_multiple_gpu_cards).

Illustration from the TensorFlow tutorial:

![](https://www.tensorflow.org/images/Parallelism.png)

### Batch size / sub-batch size

[As explained by Tim Dettmers](http://timdettmers.com/2014/10/09/deep-learning-data-parallelism/), in order not to waste GPU resources batch size for one GPU should an integer multiple of 32 and at least 64. When we use multiple GPUs we should keep the size of each sub-batch the same, thus the total mini-batch size can be computed as `batch_size = gpu_count * sub_batch_size`. For example for 2,4 or 8 GPUs we should have batch size of at least 128, 256 or 512 respectively. For more GPUs the mini-batch may get quite big, which would normally slow down the convergence. Adjusting learning rate as in [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677) might help. Also we should make sure we can generate/provide data samples fast enough.

### Advanced algorithms

We might try to reduce the communication.

[1-Bit Stochastic Gradient Descent and Application to Data-Parallel Distributed Training of Speech DNNs](https://www.microsoft.com/en-us/research/publication/1-bit-stochastic-gradient-descent-and-application-to-data-parallel-distributed-training-of-speech-dnns/) ( F. Seide, Hao Fu, Jasha Droppo, Gang Li, and Dong Yu, "1-bit stochastic gradient descent and its application to data-parallel distributed training of speech DNNs," in Proceedings of Interspeech, 2014.)

- gradients quantized from float32 to 1 bit
- residual is added to the next mini-batch
- warm-up phase without parallelism
- automatic minibatch scaling

#### Block-momentum

K. Chen and Q. Huo, "Scalable training of deep learning machines by incremental block training with intra-block parallel optimization and blockwise model-update filtering," in Proceedings of ICASSP, 2016.

## Implementations

### Pure TensorFlow

- https://www.tensorflow.org/tutorials/deep_cnn#training_a_model_using_multiple_gpu_cards
- https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py

https://github.com/tensorflow/benchmarks/blob/master/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py

#### How it works?

- device placement using `with tf.device(...)`
- CUDA_VISIBLE_DEVICES

#### Where to put parameters?

- CPU
- one of GPUs

### Keras + TensorFlow

Let's break the backend transparency and use TF operations explicitly.

- fcholet:
  - https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html#multi-gpu-and-distributed-training
  - just a sketch
- https://github.com/kuza55/keras-extras
  - https://medium.com/@kuza55/transparent-multi-gpu-training-on-tensorflow-with-keras-8b0016fd9012
  - still uses `feed_dict`
- https://github.com/avolkov1/keras_experiments
  - claims to support TF queues and even NCCL

How to combine?

- Keras for both creating model and training with all the bells and whistles (callbacks, etc.)
- Keras only for creating model, pure TF for training
- vanilla Keras `Optimizer` only supports one loss
  - need to modify to perform gradient averaging

### Keras + MXNet

- https://devblogs.nvidia.com/parallelforall/scaling-keras-training-multiple-gpus/
- https://medium.com/@julsimon/apache-mxnet-support-in-keras-83de7dec46e5

- CIFAR 10
- they use queues, multi-GPU and preprocessing from MXNet
- (+) they show very good scaling
  - on high end hardware - NVIDIA-DGX1 with 4x P100 and NVLink
  - what about cloud instances or commodity hardware?
- (-) cannot be used when our project already depend on TensorFlow

### CNTK

- https://docs.microsoft.com/en-us/cognitive-toolkit/Multiple-GPUs-and-machines
- https://docs.microsoft.com/en-us/cognitive-toolkit/Enabling-1bit-SGD

#### Keras + CNTK

- https://docs.microsoft.com/en-us/cognitive-toolkit/Using-CNTK-with-Keras
- integration with Keras is not complete
- no multi-GPU support yet for this combination

### PyTorch

http://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html

## Measurements & results

- throughput (samples/sec), time per batch
- TensorBoard
- nvprof

## Our attempt

Inspired on kuza55 `make_parallel()`. In the first observation kuza55/make_parallel model appeared to have gradients computed on a single device. Also it does not perform true gradient averaging. So I decided to try implementing that in another way.

- https://github.com/fchollet/keras/issues/7515
- https://github.com/rossumai/keras-multi-gpu

TODO: explain the approach

After I examined the TF graph of kuza55/make_parallel closely in TensorBoard and after Keras 2.0.7 fixed the naming of tensors so that they're properly grouped in TensorBoard it became clear that the gradients are indeed distributed.

There's a problem I encountered in my implementation. Placeholders for inputs of targets and sample_weights are created in the tower models, but are not used. Thus the trainign fails at runtime.

## What appeared in the meanwhile

https://github.com/avolkov1/keras_experiments

## Roadmap

### Data prefetching via TF queues

The first problem is that training with vanilla Keras/TensorFlow is not efficient even on a single GPU. Due to providing data via `feed_dict` there's no overlap between memcpy/compute operations and the GPU cannot be fully utilized for computation.

This could be solved by prefetching data asynchronously via TF queues. At the moment computation of a batch begins the data is already there. It may significantly improve performance on a single GPU.

With this done the approach of `kuza55/make_parallel` might be viable as it is.

For multi-GPU setup, this approach would solve one more problem of `kuza55/make_parallel`. Now the whole mini-batch is sent to all GPU and then sliced. Thus `(N-1)/N` sub-batches are transferred in vain. By taking individual sub-batches from the queue we would transfer only necessary data.

The rest of changes may further improve performance but probably are not vital.

### Gradient averaging

So far `kuza55/make_parallel` just computes predictions

- does proper gradient averaging instead of

### Optimized communication with NCCL operations

- reduce/transfer gradients and broadcast weights via NCCL operations

### Quantized gradients

A limiting factor is exchanging gradients and weights between GPUs or with CPU.

- try to reduce amount of transfered data by 1-bit SGD

### Better learning rates for bigger batch sizes

- use techniques from [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677) to handle larger mini-batch sizes
  - learning rate scaled up
  - warm-up phase with lower learning rate

### Constraints on the model usage

- multi-GPU only for training
- we need to be able to save/load the weights

## Conclusion
