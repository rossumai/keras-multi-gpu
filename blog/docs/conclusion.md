# Conclusion

What remains to efficient multi-GPU data-parallel training in Keras over TensorFlow?

- seems necessary:
    - using a queue for asynchronous data feeding:
        - mainly double-buffering at the GPU side
    - possibly a queue for feeding data into the TF memory
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
- NCHW data format
- not necessary to explicitly perform gradient averaging
    - sufficient to just concat the outputs
    - collocate_gradients_with_ops = True
- CNN with at least 500k parameters
- dataset like CIFAR10 or larger
- we have to check that the converge same as serial model

Possible to use some newly published packages at the expense of more complicated API:

- horovod
- tensorpack
