# Algorithms and techniques

## Basic algorithm - SGD

Let's quickly review the basic algorithm for training deep-learning models - Stochastic Gradient Descent (SGD).

Roughly said: In our model we have weights (parameters) and loss function. In each step we want to update the weights so that the loss function decreases. To do that we compute gradient of the loss (vector of partial derivatives of loss with respect to each weight) using back-propagation. We apply the gradient using an update rule (specific to the particular optimizer) to the weights. We iterate these steps until convergence.

Two extreme variants are full batch GD (gradient computed using all the training data) and pure SGD (gradient approximated from one data sample at time). In practice mini-batch SGD iterates small random subsets (batches) of training set. It's a trade-off to fit batches within GPU memory and approximate gradient well enough.

For GPU training the weights can be stored on the GPU.

## Ways of parallelism in deep learning

There are two principial ways how to parallelize training of deep learning models:

- *data parallelism* - parts of data mini-batch are distributed over GPUs and trained on the same model
- *model parallelism* - parts of the model are distributed over GPUs and trained on the same mini-batch of data

Nice overview is an a series of articles by Tim Dettmers: [Data Parallelism](http://timdettmers.com/2014/10/09/deep-learning-data-parallelism/), [Model Parallelism](http://timdettmers.com/2014/11/09/model-parallelism-deep-learning/). It's recommended to use data parallelism for convolutional and recurrent networks with a lot of computation and not too much parameters, while model parallelism for networks with too many parameters that would not fit within a single GPU memory (eg. layers of LSTM in GNMT).

Since our motivation in Rossum is to speed up training of some bigger convolutional image models in this article we will focus only on data-parallelism.

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

### Data transfers

- in single-GPU training
    - transfering batches to GPU (host to device)
    - computing gradients (device to device)
  - occasionally transfering weights for checkpointing (device to host)
- in multi-GPU training
    - transfering batches to each GPU
    - transfering gradients
    - transfering weights

Eg. for parameter server on CPU:

- batches to each GPU (host to device)
- computing gradients (device to device)
- all gradients to PS (device to host)
- gradients aggregated and weights updated
- weights to each GPU (host to device)

### Asynchronous data feeding

In a basic case in each batch we perform these steps:

- copy mini-batch from CPU to GPU (data transfer)
- compute gradients and update weights (compute)

If we trivially run both kind of operations sequentially, it's obvious that while transfering data, computation cannot run and we underutilize the GPU. In case we have a lot of data and/or low bandwidth, data transfer time might dominate over the compute time.

A solution to that is feeding the data asynchronously.

- multiple CUDA streams - one for feeding, one for running CUDA kernels
- there's a queue of batches being copied to the GPU buffer
    - at least two batches (one being computed, one being copied)
    - like double buffering in classic graphics rendering pipeline, except the direction of data is opposite
    - computation takes data that's already present in GPU memory
    - during computation another batch is being copied to the GPU memory
- it requires that each batch can be prepared through the whole pipeline faster than the time for the computation

#### TensorFlow queues

- seems to perform only async feeding into TF memory on CPU

#### StagingArea

- seems to really implement a GPU-resident queue and can do double buffering

#### Dataset API

- ???

### Batch size / sub-batch size

[As explained by Tim Dettmers](http://timdettmers.com/2014/10/09/deep-learning-data-parallelism/), in order not to waste GPU resources batch size for one GPU should an integer multiple of 32 and at least 64. When we use multiple GPUs we should keep the size of each sub-batch the same, thus the total mini-batch size can be computed as `batch_size = gpu_count * sub_batch_size`. For example for 2,4 or 8 GPUs we should have batch size of at least 128, 256 or 512 respectively. Too big batch size will result in out-of-memory error (OOM). Also we should make sure we can generate/provide data samples fast enough.

The naming is not consistent, so you can see:

- batch size / global batch size - for N GPU in total
- sub-batch size / batch size - one one GPU

### Better learning rates for bigger batch sizes

For more GPUs the mini-batch may get quite big, which would normally slow down the convergence. Adjusting learning rate as in [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677) might help.L earning rate scaled up with according to batch size. In order to keep convergence there's a warm-up phase with lower learning rate.

### Where to put parameters (weights)?

[TF High-performance models - Variable Distribution and Gradient Aggregation](https://www.tensorflow.org/performance/performance_models#variable_distribution_and_gradient_aggregation)

There are two main options for location of weights:

- parameter server (PS)
    - one device holds a copy where updates are done
    - the updated weights are distributed to all GPUs (and used for next computation)
- replicated parameters
    - each GPU has its copy and performs updates itself
    - weights need not be transferred
        - except for occasional checkpointing
    - only gradients are transferred between GPUs

#### Parameter server

- PS device can typically be:
  - CPU (RAM on the computer)
  - GPU (RAM on one of the GPUs)

There are two main options where to place the network parameters. In most tutorials people recommend to put them on CPU. The reason is that there's only communication between CPU and GPU. Also the memory allocation on GPUs should be then symmetric.

In this case sub-batch gradients from each GPU are copied to CPU, weights updated and copied to each GPU.

When the weights are placed on GPU and then distributed to other GPUs in the worst case the communication may go through the CPU (ie. GPU->CPU + CPU->GPU).  On the other hand in practice I saw peer-to-peer memcpy operations, ie. data is transferred between GPUs directly through the PCIe bus. Anyway a single GPU may have less aggregate bandwidth than CPU to communicate with all other GPUs.

It's recommended to try both options and measure what's better for particular model architecture or hardware.

### How to perform data transfers?

No matter if we use parameter server or replicated mode, we have to transfer gradients and possibly weights. There are two options:

- implicit copy
- NCCL
    - [NVIDIA blog: Fast Multi-GPU collectives with NCCL](https://devblogs.nvidia.com/parallelforall/fast-multi-gpu-collectives-nccl/)
    - [TF performance models - about NCCL](https://www.tensorflow.org/performance/performance_models#replicated_variables_in_distributed_training#nccl)

By default TensorFlow performs implicit copy. In most cases (few GPUs) this is the fastest option. Another option is NCCL, an NVIDIA's optimized implementation of collective operations, such as broadcast, scatter, gather or aggregate. It's available in TensorFlow in [tf.contrib.nccl](https://www.tensorflow.org/api_docs/python/tf/contrib/nccl). Also TensorFlow reports it can be useful for 8 or more GPUs, so probably for simpler setups increased complexity is not worth it.

### Data format NCHW vs. NHWC

[TF Performance guide - Data formats](https://www.tensorflow.org/performance/performance_guide#data_formats)

Ordering of dimensions in the data format for image data matters for GPU training. There are two conventions:

- `NCHW` or `channels_first`
- `NHWC` or `channels_last`

cuDNN is optimized for NCHW, TensorFlow and Keras support both (`NHWC` is default in TF). In Keras it's possible to set it in [`keras.json`](https://keras.io/backend/#kerasjson-details) config file.

## Conclusion

For training our bigger convolutional network models we can utilize data-parallelism, ie. replicate the network over multiple GPUs and train on multiple mini-batches in parallel. The safest bet is to put the weights on cpu in parameter server mode. In this case we scatter a batch to GPUs, compute gradients, aggregate them, update weights and broadcast them. We should scale the batch size and learning date by number of GPUs. Input data has to be provided asynchronously through the pipeline, so that GPU computation does not wait. Data can be in for NCHW format to get bonus performance from cuDNN.
