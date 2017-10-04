# Measurements

[Big sheet with measurments](https://docs.google.com/spreadsheets/d/1c5yGydEANMzHjBufTzph0w-WGwJyiwPMRYz3yBZatb4/edit)

- questions
- what to measure:
    - throughput (samples/sec) - used in TF benchmarks
        - mean or possibly median over all batches
    - time per epoch - too rough
    - scaling of throughput (speed-up)
        - `speedup_N = thoughput_N / thoughput_1`
            - we want it to approach `N`
    - percentage of real speed-up to ideally linear (normalized speedup)
        - `speedup_percentage = 100 * speedup_N / N` (%)
            - ideally we want it to approach 100%
- warm-up
    - it's observed that training is slow at the beginning and then after some warm-up period it stabilizes at much faster speed
    - so we can ignore data from the warm-up period
    - TODO: make a graph in time
        - how to quantify the warm-up period - time, number of epochs/batches?
- TensorBoard
    - how are operations (especially gradients) distributed to devices?
    - logging in TensorBoard format, opening the logs in the web app
- nvprof
    - allows to see timeline of low-level CUDA operations!
    - we can see it to analyze the overhead of data transfers, eg. with sync feeding
    - more useful information: compute utilization, bandwidth
    - modes of operations:
        - save the logs for GUI
        - print summary stats of operations to stdout
            - useful to see quickly if I/O dominates
        - print all operations to stdout
    - can be visualized in a Eclipse-based GUI
    - logs can be big (100MB+), SQLite format
    - needs libcupti
    - possible/necessary to cut a small window while importing to the GUI
    - would be nice to make a tool to analyze/preprocess the SQLite outside the GUI
- TensorFlow profiler
    - profiling at the level of TF operations
    - needs libcupti
        - a bit tricky to install on Ubuntu - we need to manually install a package from newer distribution version

## What is some good architecture and dataset to test scalable training?

Architecture:

- inception3
- resnet50

Datasets:

- imagenet-synth
- cifar10

## What kind of variable update for multiple GPUs?

- parameter server on CPU - consistently the best
- parameter server on GPU - sometimes better for 2 GPUs
- replicated
- replicated with NCCL - faster than replicated for many GPUs

## What data format: NCHW vs. NHWC?

NCHW (channels_first) consistently faster (10-25 % in TensorFlow benchmarks).

Data format doesn't have much effect on scaling - NCHW work better in any case.

## Is 7gforce usable for multi-GPU training or do we have to use cloud instances?

- 7gforce:
  - best scaling: 4.487x
    - 6 GPUs on inception3/imagenet-synth in tf_cnn_benchmarks
  - least overhead: 80.75% of ideal scaling at 1.615x speed-up
    - for 2 GPUs with on inception3/imagenet-synth in tf_cnn_benchmarks
  - otherwise rather poor scaling
- az-2x-m60:
  - almost 2x scaling for 2 GPUs on inception3/imagenet-synth and many models on cifar10 in tf_cnn_benchmarks

## Is gradient averaging necessary or can we just concat predictions?

- TF performs gradient averaging
- kuza55 just concatenates predictions + colocate_gradients_with_ops

## How different architectures are suitable to data/model-parallelism?

- dense layers
- convolutional layers
- recurrent layers

## What properties of datasets make them suitable for parallel training?

- size of images?
- number of data samples?
