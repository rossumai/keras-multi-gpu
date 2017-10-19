# [WIP] keras-multi-gpu

Multi-GPU data-parallel training in Keras

Keras issue: [#7515 Correct data-parallel SGD implementation in Keras](https://github.com/fchollet/keras/issues/7515)

The goal is to implement data-parallel multi-GPU training with gradient-averaging properly in Keras (at least explicitly for TensorFlow backend).

In this issue I'd like to discuss a particular approach which tries to fix problems of current solutions. Since Keras seems not to be designed for data-parallel SGD, I'm trying to find ways how to modify or adapt the Keras code, while keeping the API philosophy. Since this problem is quite important for  many people, including our team at @rossumai, I'd like to ask for advice. Any feedback is really welcome.

## Quick outline of the data-parallel SGD algorithm

We use N identical model replicas (towers) to train on slices on a mini-batch. Model parameters are placed on a parameter server device (PS), CPU or one of the GPUs, computations are made on N worker devices. A minibatch of inputs is split into N sub-batches, distributed to each worker which computes the forward and backwards pass, the resulting gradients are sent to the PS device, averaged and used to update the weights, which are then copies back to the workers.

## Previous experiments

As a baseline I checked the [TensorFlow CIFAR 10 multi-GPU tutorial](https://www.tensorflow.org/tutorials/deep_cnn#training_a_model_using_multiple_gpu_cards). It worked as expected for 1-4 GPUs (TensorFlow 1.2.1, CUDA 8.0, GTX 1070).

I tried the approach of [kuza55/keras-extras](https://github.com/kuza55/keras-extras), discussed earlier in other issues (#2436) and blog post [Transparent Multi-GPU Training on TensorFlow with Keras](https://medium.com/@kuza55/transparent-multi-gpu-training-on-tensorflow-with-keras-8b0016fd9012), adapting MNIST MLP and CIFAR10 Keras examples (Keras 2.0.6, TensorFlow 1.2.1, CUDA 8.0). In practice using more than one GPU lead to decrease of performance. Between 2 and 4 GPUs there was a 2x speedup, however.

https://gist.github.com/bzamecnik/390d3802b31ce766fbf6fd6c9fd682d3

## Problems with kuza55/keras-extras

After examining the graph in TensorBoard I discovered a problem in this approach: gradients are not computed in parallel on each device, but in whole on the parameter service device. Indeed each worker computes only the predictions which are distributed to PS and concatenated. The loss + gradients are computed there. Another potential problem is that the whole mini-batch is fed to each device which only takes it's slice. We waste our precious IO bandwidth.

## Proposed fixes

* gradients should be computed in each tower separately, then averaged on PS device
* only tower sub-batch slices of input/labels transferred to each tower (not full batch)
* should we use queue for providing inputs asynchronously?
* are the model parameters properly placed on the PS device and shared?
* in addition for a correct parallel SGD implementation we should incorporate corrections outlined in the [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677) paper:
    * scaled learning rate with warm-up
    * momentum correction
    * batch normalization correction
    * gradient aggregation
        * for averaging we can put all the normalization terms inside the loss and then reduce by sum
        * "Normalize the per-worker loss by total minibatch size kn, not per-worker size n."
    * random shuffling: "Use a single random shuffling of the training data (per epoch) that is divided amongst all k workers."
    * gradients aggregatin should be done in parallel with back-propagation
        * "As soon as the gradient for a layer is computed, it is aggregated across workers, while gradient computation for the next layer continues."

## Proposed implementation

### Parallel losses and gradients - DataParallelOptimizer

Since the Keras API is at present not directly suitable for data-parallel SGD computation, in the first step of making a working prototype we can make different implementations of Optimizer and Model, let's say DataParallelOptimizer and DataParallelModel.

We need to compute loss and gradients in parallel. Tensor for loss is created by Keras within `Model.compile()` and stored as `Model.total_loss`. Gradients are computed in `Optimizer.get_gradients()` which is called in lazily created functions `Model.{train,test,predict}_function()` (called from `fit()`, etc.). This function accepts single loss tensor. Various optimizers then compute updates based on the gradients. The problem is a single loss tensor (which can be placed on one device) passed to  `Optimizer.get_updates()`.

So far the only way I see it so change `Model.total_loss` from a single tensor into a list of tensors, each of them able to be placed on a different device.
`DataParallelOptimizer` wrapper class can derive from `Optimizer` and override `get_gradients()` to accept loss as a list of tensors and average them. This would be the place of synchronization of the GPU workers. The `get_updates()` function (implemented any of the wrapped `Optimizer`) just calls `get_gradients()`. Note that thanks to the option `collocate_gradients_with_ops=True` in TF implementation of `K.gradients()` the gradient ops would automatically be placed on the same device as the loss ops even though it's called outside compile() and the device scope. (TODO: issue link)

### Model replication and feeding data - DataParallelModel

We need a Model which contains the replicas (towers) and provides the list of losses to the `DataParallelOptimizer`. We would adapt the code in the `make_parallel()` function from kuza55/keras-extras. The `DataParallelModel` would take via the constructor an instance of the basic model. Similar like in `make_parallel()` it would make N replicas of this model placed on different devices. We could try to set TF variable reuse after the first replica. Also we make the merged model, which concatenates the outputs, and use it for the actual training. Better then slicing the duplicate inputs we can pass the sub-batch inputs separately and route them to each replicate directly.

Then it would override `compile()` which has to call `compile()` on each replica (and the merged model) - in order to place losses and gradients - and gather `total_loss` operation from all replicas. In `compile()` we also wrap the provided optimizer with `DataParallelOptimizer` and inject both the total_loss list and the DataParallelOptimizer instance to the merged model. The rest of the methods in `DataParallelModel` will be proxied to the merged model.

In case we want to avoid slicing the inputs we could change the inputs in {train,test,predict}_function() and perform the slice in `*_on_batch()` functions.

## Code

https://github.com/rossumai/keras-multi-gpu

I have prepared an implementation of `DataParallelOptimizer` and I'm working on DataParallelModel. The mechanism of the latter is not as clear at the moment. In the first stage I'd like to make a working prototype, then make experiments to show that the model produces correct results and that we obtain benefit from scaling to multiple GPUs. Next I wish to make the API cleaner. So far I think the code might be separate from Keras, since it will depend on TensorFlow explicitly and I'm not sure about Theano support.

If you read this rather longer I'd like to kindly ask for advice if you think this approach is feasible or you see any problems with that. Any feedback is really welcome. Thanks!
