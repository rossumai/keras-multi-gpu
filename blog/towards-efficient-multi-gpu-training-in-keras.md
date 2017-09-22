# [WIP] Towards efficient multi-GPU training in Keras over TensorFlow

In this article I'd like to summarize the current state of attempts for data-parallel multi-GPU training in Keras over TensorFlow and a roadmap of steps to make it efficient and usable in practice.

We will cover two problems: efficiently training on a single GPU and extending training to multiple GPUs on the same machine.

We started our experiments in July 2017 and in the meanwhile there appeared some open-source experiments that seem to solve the scaling problem, either on different backend (MXNet, CNTK) or even using TensorFlow.

So let's try to dive in the problem, see the landscape of of solutions and review a bunch of measurements in practice to see what's working and what doesn't.

## Introduction

### Why?

It's not uncommon that bigger models take many hours to train. We already parallelize the computation by using many-core GPUs. Next step is using multiple GPUs. Although we introduce more complexity, we expect decrease in training time. Ideally the training should scale inversely and the throughput of data samples per second linearly with respect to the number of GPUs. Of course we also expect some overhead of operations that cannot be parallelized.

### Efficient GPU training with Keras + TensorFlow

https://www.tensorflow.org/performance/performance_models

### Ways of parallelism in deep learning

There are two principial ways how to parallelize traning of deep learning models:

- *data parallelism* - parts of data mini-batch are distributed over GPUs and trained on the same model
- *model parallelism* - parts of the model are distributed over GPUs and trained on the same mini-batch of data

Nice overview is an a series of articles by Tim Dettmers: [Data Parallelism](http://timdettmers.com/2014/10/09/deep-learning-data-parallelism/), [Model Parallelism](http://timdettmers.com/2014/11/09/model-parallelism-deep-learning/). It's recommended to use data parallelism for convolutional and recurrent networks with a lot of computation and not too much parameters, while model parallelism for networks with too many parameters that would not fit within a single GPU memory (eg. layers of LSTM in GNMT).

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

## Hardware

- connection of GPUs to CPU
  - PCIe
  - at best 16x lane,
  - multiple GPUs
- interconnection of GPUs
  - PCIe
    - on commodity hardware
  - NVLink
    - GPUs connected in a hyper-cube topology
    - on high-end hardware such as NVIDIA DGX-1
    - higher performance

[List of common GPUs for deep learning and their parameters](https://docs.google.com/spreadsheets/d/1pQNWTLfsBmclB3lojc5DHPZCRoKATjNqDA1Sbwq4zeE/edit#gid=0).

### Comparison of a custom 7-GPU machine with cloud instances

- 7-GPU - 6x 1070 + 1x 1080
  - too many devices share limited number of PCIe lanes
  - too slow host-device connection even for one device
  - in practice doesn't work well for multi-GPU training
- Azure Standard_NV12 (2x Tesla M60)
  - seems to work well
- Azure Standard_NV24 (4x Tesla M60) - TOOD
- HW with NVLink (like NVIDIA DGX-1) would be much better
  - we don't have access to any -> can't measure

### Topology

Tip from https://github.com/BVLC/caffe/blob/master/docs/multigpu.md

```
$ nvidia-smi topo -m
```

```
Legend:

  X   = Self
  SOC  = Connection traversing PCIe as well as the SMP link between CPU sockets(e.g. QPI)
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe switches (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing a single PCIe switch
  NV#  = Connection traversing a bonded set of # NVLinks
```

7gpu - two groups (within group just PCIe switch, between via CPU):
```
        GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    CPU Affinity
GPU0     X      PIX     PIX     PHB     PHB     PHB     PHB     0-19
GPU1    PIX      X      PIX     PHB     PHB     PHB     PHB     0-19
GPU2    PIX     PIX      X      PHB     PHB     PHB     PHB     0-19
GPU3    PHB     PHB     PHB      X      PIX     PIX     PIX     0-19
GPU4    PHB     PHB     PHB     PIX      X      PIX     PIX     0-19
GPU5    PHB     PHB     PHB     PIX     PIX      X      PIX     0-19
GPU6    PHB     PHB     PHB     PIX     PIX     PIX      X      0-19
```

Azure 2x M60 - two GPUs on one board (nice):
```
GPU0    GPU1    CPU Affinity
GPU0     X     SOC    0-11
GPU1    SOC     X     0-11
```

TODO: measurements
- bandwidth
  - CUDA samples benchmark, nvprof from a training run
- training experiments

```
cd /usr/local/cuda/samples/1_Utilities/bandwidthTest
sudo make
./bandwidthTest
```

Azure 2x M60:
```
[CUDA Bandwidth Test] - Starting...
Running on...

 Device 0: Tesla M60
 Quick Mode

 Host to Device Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)	Bandwidth(MB/s)
   33554432			6070.1

 Device to Host Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)	Bandwidth(MB/s)
   33554432			6536.0

 Device to Device Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)	Bandwidth(MB/s)
   33554432			133094.5

Result = PASS
```

## Implementations

### Pure TensorFlow

- https://www.tensorflow.org/tutorials/deep_cnn#training_a_model_using_multiple_gpu_cards
- https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py

https://github.com/tensorflow/benchmarks/blob/master/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py

#### How it works?

TensorFlow allows to place operation (nodes of the computational graph) to be placed on differnet available devices, either automatically or manually.

[TensorFlow tutorial: Using GPUs - Manual device placement](https://www.tensorflow.org/tutorials/using_gpu#manual_device_placement)

Devices are named `/cpu:0` (CPU), `/gpu:0`, `/gpu:1`, ... (visible GPUs).

##### Device placement

Placing operation on devices is done using device scopes that are implemented in Python as context managers.

You can explicitly place an operation on device `/gpu:1` like this:

```
with tf.device("/gpu:1"):
  # ... some tf operation ...
```

In order to check it works well it's possible to [log device placement](https://www.tensorflow.org/tutorials/using_gpu#logging_device_placement) or check the graph in TensorBoard.

##### Device visibility

In case the machine has multiple devices it's possible to restrict devices visible to a process by setting environment variable `CUDA_VISIBLE_DEVICES` with the list of device numbers. You can either provide it in bash or in Python.

```
$ CUDA_VISIBLE_DEVICES=0,1,4,5 python train.py
```

The devices in this order are then mapped in TF to `/gpu:0`, ...

#### Where to put parameters?

CPU, one of GPUs or all GPUs?

There are two main options where to place the network parameters. In most tutorials people recommend to put them on CPU. The reason is that there's only communication between CPU and GPU. Also the memory allocation on GPUs should be then symmetric.

When the weights are places on GPU and then distributed to other GPUs in the worst case the communication may go through the CPU (ie. GPU->CPU + CPU->GPU). On the other hand in practice I saw peer-to-peer memcpy operations, ie. data is transferred between GPUs directly through the PCIe bus.

I'd recommend to try both options and measure what's better.

Besides this 'parameter server' mode also possible to 'replicate' variables to all GPUs:

https://www.tensorflow.org/performance/performance_models#variable_distribution_and_gradient_aggregation

### Keras + TensorFlow

Let's break the backend transparency and use TF operations explicitly.

#### How to combine Keras and TensorFlow?

- Keras for both creating model and training with all the bells and whistles (callbacks, etc.)
- Keras only for creating model, pure TF for training
- vanilla Keras `Optimizer` only supports one loss
  - need to modify to perform gradient averaging

#### Existing attempts

- fcholet (Keras author):
  - introducing TensorFlow backend - [Keras as a simplified interface to TensorFlow: tutorial](https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html#multi-gpu-and-distributed-training)
  - just a code sketch of indended use in model parallelism and data parallelism
- keras/issues:
  - [Does Keras support using multiple GPUs? #2436](https://github.com/fchollet/keras/issues/2436)
  - [Data parallelism with multiple GPUs with fit_generator #7698](https://github.com/fchollet/keras/issues/7698)
  - [[WIP] Correct data-parallel SGD implementation in Keras #7515](https://github.com/fchollet/keras/issues/7515)
- https://github.com/kuza55/keras-extras
  - blog: [Transparent Multi-GPU Training on TensorFlow with Keras
](https://medium.com/@kuza55/transparent-multi-gpu-training-on-tensorflow-with-keras-8b0016fd9012)
  - distilled from the forums snippets
  - still uses `feed_dict`
  - some users claim to see speedup
  - not packaged yet
- https://github.com/avolkov1/keras_experiments
  - claims to support TF queues and even NCCL!
  - users report it scales well
  - quite rough prototype, no python package yet
  - so far the best starting ground

### Keras + MXNet

While MXNet backed appeared in Keras in May 2017, multi-GPU support of Keras over MXNet appeared recently: [https://devblogs.nvidia.com/parallelforall/scaling-keras-training-multiple-gpus/](Scaling Keras Model Training to Multiple GPUs) in NVIDIA blog (2017-08-16).

- dataset: ImageNet (250 GB), model: ResNet-50
  - should be enough to benefit from multi-GPU training
- they use queues, multi-GPU and preprocessing from MXNet
- (+) details are hidden behind quite nice API
- (+) they show very good scaling
  - on high end hardware - NVIDIA-DGX1 with 4x P100 and NVLink
  - what about cloud instances or commodity hardware?
- (-) cannot be used when our project already depend on TensorFlow

### CNTK

Pure CNTK claims to support multi-GPU training, including algorithms to reduce communication overhead like 1-bit SGD.

- https://docs.microsoft.com/en-us/cognitive-toolkit/Multiple-GPUs-and-machines
- https://docs.microsoft.com/en-us/cognitive-toolkit/Enabling-1bit-SGD

#### Keras + CNTK

- https://docs.microsoft.com/en-us/cognitive-toolkit/Using-CNTK-with-Keras
- [integration with Keras is not complete](https://docs.microsoft.com/en-us/cognitive-toolkit/Using-CNTK-with-Keras#known-issues)
- [multi-GPU support appeared just recently](https://docs.microsoft.com/en-us/cognitive-toolkit/Using-CNTK-MultiGPU-Support-with-Keras) (2017-09-15)
  - it explicitly uses CNTK facilities for setup, but then Keras's `Model.fit()`
  - some code examples available but not a complete script
- (-) cannot be used when our project already depend on TensorFlow

TODO: run, measure, review

### PyTorch

http://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html

TODO: run, measure, review

## Measurements & results

- throughput (samples/sec), time per batch
- TensorBoard
  - how are operations distributed to devices?
- nvprof
  - compute utilization
  - bandwidth

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

This could be solved by prefetching data asynchronously via TF queues. At the moment computation of a batch begins the data is already there. It may significantly improve performance on a single GPU. There's a [basic queue API](https://www.tensorflow.org/api_guides/python/reading_data#Reading_from_files) for and a currently recommended [Dataset API](https://www.tensorflow.org/programmers_guide/datasets).

With this done the approach of `kuza55/make_parallel` might be viable as it is.

For multi-GPU setup, this approach would solve one more problem of `kuza55/make_parallel`. Now the whole mini-batch is sent to all GPU and then sliced. Thus `(N-1)/N` sub-batches are transferred in vain. By taking individual sub-batches from the queue we would transfer only necessary data.

The rest of changes may further improve performance but probably are not vital.

### Gradient averaging

So far `kuza55/make_parallel` just computes predictions

- does proper gradient averaging instead of concatenating the outputs

### Optimized communication with NCCL operations

Possible to reduce/transfer gradients and broadcast weights via NCCL operations.

TensorFlow already supports that in [tf.contrib.nccl](https://www.tensorflow.org/api_docs/python/tf/contrib/nccl).

In the TensorFlow [High-Performance Models] tutorial they advise that NCCL may help with 8 or more GPUs, however for less GPU the plain implicit copy performs better.

So it's an option for performance improvement at the cost of higher complexity and it has to be measure if it helps in a particular case or not.

### Quantized gradients

A limiting factor is exchanging gradients and weights between GPUs or with CPU.

- try to reduce amount of transfered data by 1-bit SGD

Nice to have. Doesn't seem to be implemented in Keras/TF so far.

### Better learning rates for bigger batch sizes

- use techniques from [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677) to handle larger mini-batch sizes
  - learning rate scaled up
  - warm-up phase with lower learning rate

### Constraints on the model usage

- multi-GPU only for training
- we need to be able to save/load the weights
- nice API

## Conclusion

TODO
