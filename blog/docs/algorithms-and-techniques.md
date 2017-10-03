# Algorithms and techniques

## Ways of parallelism in deep learning

There are two principial ways how to parallelize traning of deep learning models:

- *data parallelism* - parts of data mini-batch are distributed over GPUs and trained on the same model
- *model parallelism* - parts of the model are distributed over GPUs and trained on the same mini-batch of data

Nice overview is an a series of articles by Tim Dettmers: [Data Parallelism](http://timdettmers.com/2014/10/09/deep-learning-data-parallelism/), [Model Parallelism](http://timdettmers.com/2014/11/09/model-parallelism-deep-learning/). It's recommended to use data parallelism for convolutional and recurrent networks with a lot of computation and not too much parameters, while model parallelism for networks with too many parameters that would not fit within a single GPU memory (eg. layers of LSTM in GNMT).

### Limiting the scope

Since our motivation in @rossumai is to speed up training of some bigger convolutional image models in this article we will focus only on data-parallelism. Also we restrict the solution to Keras over TensorFlow backend. Also we will not consider training distributed to mulitple machines.

## Algorithms for data-parallel training

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

## Advanced algorithms

We might try to reduce the communication.

[1-Bit Stochastic Gradient Descent and Application to Data-Parallel Distributed Training of Speech DNNs](https://www.microsoft.com/en-us/research/publication/1-bit-stochastic-gradient-descent-and-application-to-data-parallel-distributed-training-of-speech-dnns/) ( F. Seide, Hao Fu, Jasha Droppo, Gang Li, and Dong Yu, "1-bit stochastic gradient descent and its application to data-parallel distributed training of speech DNNs," in Proceedings of Interspeech, 2014.)

- gradients quantized from float32 to 1 bit
- residual is added to the next mini-batch
- warm-up phase without parallelism
- automatic minibatch scaling

### Block-momentum

K. Chen and Q. Huo, "Scalable training of deep learning machines by incremental block training with intra-block parallel optimization and blockwise model-update filtering," in Proceedings of ICASSP, 2016.

## Techniques

### Batch size / sub-batch size

[As explained by Tim Dettmers](http://timdettmers.com/2014/10/09/deep-learning-data-parallelism/), in order not to waste GPU resources batch size for one GPU should an integer multiple of 32 and at least 64. When we use multiple GPUs we should keep the size of each sub-batch the same, thus the total mini-batch size can be computed as `batch_size = gpu_count * sub_batch_size`. For example for 2,4 or 8 GPUs we should have batch size of at least 128, 256 or 512 respectively. Too big batch size will result in out-of-memory error (OOM). Also we should make sure we can generate/provide data samples fast enough.

### Better learning rates for bigger batch sizes

For more GPUs the mini-batch may get quite big, which would normally slow down the convergence. Adjusting learning rate as in [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677) might help.L earning rate scaled up with according to batch size. In order to keep convergence there's a warm-up phase with lower learning rate.

### Where to put parameters?

- parameter server: cpu / gpu
- replicated parameters
  - implicit copy vs. NCCL

CPU, one of GPUs or all GPUs?

There are two main options where to place the network parameters. In most tutorials people recommend to put them on CPU. The reason is that there's only communication between CPU and GPU. Also the memory allocation on GPUs should be then symmetric.

When the weights are places on GPU and then distributed to other GPUs in the worst case the communication may go through the CPU (ie. GPU->CPU + CPU->GPU). On the other hand in practice I saw peer-to-peer memcpy operations, ie. data is transferred between GPUs directly through the PCIe bus.

I'd recommend to try both options and measure what's better.

Besides this 'parameter server' mode also possible to 'replicate' variables to all GPUs:

https://www.tensorflow.org/performance/performance_models#variable_distribution_and_gradient_aggregation

### Optimized communication with NCCL operations

Possible to reduce/transfer gradients and broadcast weights via NCCL operations.

TensorFlow already supports that in [tf.contrib.nccl](https://www.tensorflow.org/api_docs/python/tf/contrib/nccl).

In the TensorFlow [High-Performance Models] tutorial they advise that NCCL may help with 8 or more GPUs, however for less GPU the plain implicit copy performs better.

So it's an option for performance improvement at the cost of higher complexity and it has to be measure if it helps in a particular case or not.

### Data format NCHW vs. NHWC

https://www.tensorflow.org/performance/performance_guide#data_formats

Ordering of dimensions in the data format for image data matters for GPU training. There are two conventions:

- `NCHW` or `channels_first`
- `NHWC` or `channels_last`

cuDNN is optimized for NCHW, TensorFlow and Keras support both (`NHWC` is default in TF). In Keras it's possible to set it in [`keras.json`](https://keras.io/backend/#kerasjson-details) config file.
