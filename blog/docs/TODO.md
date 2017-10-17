# TODO

Measurements:

- existing Keras solutions:
  - rough numbers, basic speed-up
- baseline from TensorFlow benchmarks
  - how do high-performance models perform in practice?
  - what speed-ups are possible?
  - what are suitable models/datasets?
  - how do various configurations perform?
    - parameter server - on CPU, on GPU
    - replicated parameters - implicit copy, NCCL
    - data format: NCHW vs. NHWC
