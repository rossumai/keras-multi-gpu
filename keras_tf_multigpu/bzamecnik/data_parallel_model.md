## Our attempt

> so far in icebox, not for publishing

Inspired on kuza55 `make_parallel()`. In the first observation kuza55/make_parallel model appeared to have gradients computed on a single device. Also it does not perform true gradient averaging. So I decided to try implementing that in another way.

- https://github.com/fchollet/keras/issues/7515
- https://github.com/rossumai/keras-multi-gpu

TODO: explain the approach

After I examined the TF graph of kuza55/make_parallel closely in TensorBoard and after Keras 2.0.7 fixed the naming of tensors so that they're properly grouped in TensorBoard it became clear that the gradients are indeed distributed.

There's a problem I encountered in my implementation. Placeholders for inputs of targets and sample_weights are created in the tower models, but are not used. Thus the training fails at runtime on uninitialized variables.
