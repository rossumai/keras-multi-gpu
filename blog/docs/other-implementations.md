## Implementations

### TensorFlow

Resources:

- [Tuturial - Convolutional Neural Networks - Training a Model Using Multiple GPU Cards]( https://www.tensorflow.org/tutorials/deep_cnn#training_a_model_using_multiple_gpu_cards)
    - parameter server at CPU
    - [cifar10_multi_gpu_train.py]( https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py)
- Performance tuning:
    - [Performance Guide](https://www.tensorflow.org/performance/performance_guide)
      - [example on CIFAR10](https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10_estimator)
      - [High-Performance Models](https://www.tensorflow.org/performance/performance_models)
- [Benchmarks](https://www.tensorflow.org/performance/benchmarks)
    - [tf_cnn_benchmarks.py]( https://github.com/tensorflow/benchmarks/blob/master/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py)
        - multiple options: PS at CPU/GPU, replicated, +NCCL, distributed, etc.
        - uses StagingArea
        - it's a very imporant resource for us

#### How does it work?

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

### Keras + MXNet

While MXNet backed appeared in Keras in May 2017, multi-GPU support of Keras over MXNet appeared recently (2017-08-16) in NVIDIA blog: [Scaling Keras Model Training to Multiple GPUs](https://devblogs.nvidia.com/parallelforall/scaling-keras-training-multiple-gpus/).

- dataset: ImageNet (250 GB), model: ResNet-50
    - should be enough to benefit from multi-GPU training
- they use queues, multi-GPU and preprocessing from MXNet
- very important is to have an efficient data preprocessing pipeline
- (+) details are hidden behind quite nice API
- (+) they show very good scaling
    - on high end hardware - NVIDIA-DGX1 with 8x P100 and NVLink
    - what about cloud instances or commodity hardware?
- (-) cannot be used when our project already depend on TensorFlow
- (-) MXNet has a not so nice API compared to TensorFlow

### CNTK

Pure CNTK claims to support multi-GPU training, including synchronous and asynchronous SGD and algorithms to reduce communication overhead like 1-bit SGD or block momentum.

- [CNTK - Multiple GPUs and Machines](https://docs.microsoft.com/en-us/cognitive-toolkit/Multiple-GPUs-and-machines)
- [Enabling 1-bit SGD]( https://docs.microsoft.com/en-us/cognitive-toolkit/Enabling-1bit-SGD)
    - seems to be slightly faster than plain synchronous SGD, not too much
    - only an crude approximation of gradient
    - strange licensing stuff :)

#### Keras + CNTK

Keras supports CTNK backed since 2.0.5 (June 2017). Official docs: [Using CNTK with Keras](https://docs.microsoft.com/en-us/cognitive-toolkit/Using-CNTK-with-Keras). It seems the [integration with Keras is not complete](https://docs.microsoft.com/en-us/cognitive-toolkit/Using-CNTK-with-Keras#known-issues).

Multi-GPU support appeared just recently (2017-09-15): [Using CNTK MultiGPU Support with Keras](https://docs.microsoft.com/en-us/cognitive-toolkit/Using-CNTK-MultiGPU-Support-with-Keras).

- it explicitly uses CNTK facilities for setup, but then Keras's `Model.fit()`
- some code examples available but not a complete script
- (-) cannot be used when our project already depend on TensorFlow

### PyTorch

For inspiration about API for a data-parallel model wrapper:

- [Multi-GPU examples](http://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html)
- uses a wrapper class [torch.nn.DataParallel](http://pytorch.org/docs/master/nn.html#dataparallel-layers-multi-gpu-distributed)
- it seems that weights are replicated and gradients exchange using NCCL
- no information about using queues
- example project using this wrapper: [imagenet](https://github.com/pytorch/examples/tree/master/imagenet).
