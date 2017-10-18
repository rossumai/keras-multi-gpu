# Conclusion

We have quite extensively explored algorithms and techniques for data-parallel training of deep learning models, their implementations in Keras+TensorFlow and elsewhere. We analyzed a few existing codes for multi-GPU training in Keras. Mainly we measured a lot - we tried to establish some baseline expectations what is possible in an optimized implementation (tf_cnn_benchmarks) and which settings work in our environment. We found some combination of model architectures and datasets that can be scaled on multiple GPUs.

It seems that existing multi-GPU codes in Keras can provide speedup at some conditions and perform bad in other situations. Both models and hardware affects that. In particular a small CNN on CIFAR10 was scaling better than bigger InceptionV3 on synthetic ImageNet, in contrast to expectations. Cloud machines with good GPU connection allow good scaling. A custom machine with a lot of GPUs and very poor PCIe connection is able to scale at some conditions, but it's limited even for optimized code. Anyway due to faster GPUs it might be a viable alternative to cloud instances.

## What remains to efficient multi-GPU data-parallel training in Keras over TensorFlow?

- seems necessary:
    - asynchronous data feeding to GPU:
        - mainly double-buffering at the GPU side
    - possibly also a queue for feeding data into the TF memory
- extra performance:
    - NCHW data format for cuDNN
- nice to have:
    - learning rate according to "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour"
    - Quantized gradients

A recipe for multi-GPU training in Keras on a few GTX Pascal or Tesla M60 cards:

- parameter server at CPU
    - basic model instantiated at CPU device
- replicas at GPUs with shared variables
- slicing of input at CPU (not GPU), concatenation of outputs at CPU
- async feeding to GPU (at least double-buffering)
    - StagingArea with size 2 for each GPU
    - process:
        - before epoch: put()
        - at each batch: put() + normal training ops
        - at last batch put() some dummy batch or avoid put()
        - after epoch possibly clear()
    - tf.backend.tensorflow.Function needs patch to support extra operations
- input and targets as tf.Tensor, get() from StagingArea
- batch size - multiply base batch size by GPU count
    - as much as it fits the device divided by number of batches in GPU queue (typically 2)
- learning rate - multiply base learning rate by GPU count
    - ideally with warm-up
- multi-GPU would be used only for training, single GPU or CPU for inference
- NCHW data format for training, possibly NHWC for inference on CPU
    - thus the model should be flexible
- not necessary to explicitly perform gradient averaging
    - sufficient to just concat the outputs
    - collocate_gradients_with_ops = True
- CNN with at least 500k parameters
- dataset like CIFAR10 or larger
- we have to check that the converge same as serial model
- in order to futher reduce communication of gradients/weights we could try using quantized gradients

Possible to use some newly published packages at the expense of more complicated API:

- horovod
- tensorpack

## Further work on experiments

Since we had a limited time for the experiments and the contributions in open-source appear every day there's much more to try.

- try NCHW data format with Keras
- try using StagingArea properly (as a double buffer)
- try using TF queues properly
- profile with nvprof both tf_cnn_benchmarks and Keras models
    - visualize in NVIDIA Visual Profiler
    - look for places where computation waits for communication
        - input data, weights/gradient
- try and measure `tensorpack` and `horovod`
    - both have a little bit more complicated usage but promise really great scaling
- compare convergence of serial and parallel models
- try running the experiments on 4x M60, K80s in cloud and on other 1070s on our other machines
- try to compare speedup if data is constant in the GPU memory and we only exchange weights/gradients
- How are different architectures suitable to data/model-parallelism?
    - dense layers, convolutional layers, recurrent layers
- compare effect of slicing on CPU vs. GPU
    - hypothesis: Slicing on GPU should be slower since we transfer N batches to each GPU instead of 1. Also the maximum batch size would be limited.
- warm-up: how to quantify the warm-up period - time, number of epochs/batches?
- what kind of models can benefit from data-parallelism?
    - minimum size, minimum dataset size, ratio between conv/FC/RNN parameters?
    - how is scaling affected by data point size, number of model parameters?
