## Implementations

### TensorFlow

- https://www.tensorflow.org/tutorials/deep_cnn#training_a_model_using_multiple_gpu_cards
- https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py

- https://github.com/tensorflow/benchmarks/blob/master/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py
- https://www.tensorflow.org/performance/performance_models

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

### PyTorch

http://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html
