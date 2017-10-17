# [WIP] Towards efficient multi-GPU training in Keras over TensorFlow

In this article I'd like to summarize the current state of attempts for data-parallel multi-GPU training in Keras over TensorFlow and a roadmap of steps to make it efficient and usable in practice.

We will cover two problems: efficiently training on a single GPU and extending training to multiple GPUs on the same machine.

We started our experiments in July 2017 and in the meanwhile there appeared some open-source experiments that seem to solve the scaling problem, either on different backend (MXNet, CNTK) or even using TensorFlow.

So let's try to dive in the problem, see the landscape of of solutions and review a bunch of measurements in practice to see what's working and what doesn't.

## Introduction

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
    - claims to support TF queues (StagingArea) and even NCCL!
    - users report it scales well
    - quite rough prototype, no python package yet
    - so far the best starting ground

Suggested improvement over kuza55/avolkov1 (until using a queue):

Instead of broadcasting the full batch and then slicing, perform scatter. Otherwise we copy N-1 subbatches just to ignore them, wasting precious bandwidth.

## Roadmap

### Data prefetching via TF queues

The first problem is that training with vanilla Keras/TensorFlow is not efficient even on a single GPU. Due to providing data via `feed_dict` there's no overlap between memcpy/compute operations and the GPU cannot be fully utilized for computation.

This could be solved by prefetching data asynchronously via TF queues. At the moment computation of a batch begins the data is already there. It may significantly improve performance on a single GPU. There's a [basic queue API](https://www.tensorflow.org/api_guides/python/reading_data#Reading_from_files) for and a currently recommended [Dataset API](https://www.tensorflow.org/programmers_guide/datasets).

With this done the approach of `kuza55/make_parallel` might be viable as it is.

For multi-GPU setup, this approach would solve one more problem of `kuza55/make_parallel`. Now the whole mini-batch is sent to all GPU and then sliced. Thus `(N-1)/N` sub-batches are transferred in vain. By taking individual sub-batches from the queue we would transfer only necessary data.

The rest of changes may further improve performance but probably are not vital.

### Quantized gradients

A limiting factor is exchanging gradients and weights between GPUs or with CPU.

- try to reduce amount of transfered data by 1-bit SGD

Nice to have. Doesn't seem to be implemented in Keras/TF so far.

### Practical considerations on model usage

- multi-GPU only for training
- typically single GPU or CPU for predictions
    - NCHW vs. NHWC
- we need to be able to save/load the weights successfully
- nicer API, not hacky
