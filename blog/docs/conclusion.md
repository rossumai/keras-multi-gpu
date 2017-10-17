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



Possible to use some newly published packages at the expense of more complicated API:

- horovod
- tensorpack
