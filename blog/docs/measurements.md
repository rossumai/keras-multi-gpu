# Measurements

## What is some good architecture and dataset to test scalable training?

Architecture:

- inception3
- resnet50

Datasets:

- imagenet-synth
- cifar10

# What kind of variable update for multiple GPUs?

- parameter server on CPU - consistently the best
- parameter server on GPU - sometimes better for 2 GPUs
- replicated
- replicated with NCCL - faster than replicated for many GPUs

# What data format: NCHW vs. NHWC?

NCHW (channels_first) consistently faster (10-25 % in TensorFlow benchmarks).

Data format doesn't have much effect on scaling - NCHW work better in any case.

# Is 7gforce usable for mulit-GPU training or do we have to use cloud instances?

- 7gforce:
  - best scaling: 4.487x
    - 6 GPUs on inception3/imagenet-synth in tf_cnn_benchmarks
  - least overhead: 80.75% of ideal scaling at 1.615x speed-up
    - for 2 GPUs with on inception3/imagenet-synth in tf_cnn_benchmarks
  - otherwise rather poor scaling
- az-2x-m60:
  - almost 2x scaling for 2 GPUs on inception3/imagenet-synth and many models on cifar10 in tf_cnn_benchmarks
