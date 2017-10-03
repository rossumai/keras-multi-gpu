# TODO

Possibly split the text into multiple articles:

- overview od data-parallel training
- measurement of TensorFlow benchmarks
  - get intuition of how the various techniques work in practice
- state of efforts in Keras

Questions:

- how to speed-up long training for our models in Rossum?
  - we have available either a 7gforce machine 6x GTX 1070 + 1x GTX 1080
  - or azure 2x/4x Tesla M60
- would multi-GPU training help?
- which kind of parallelism to use?
  - data parallelism
- is it possible to use 7gforce machine given it's low host-device bandwidth?
- what kind of models can benefit from data-parallelism?
  - minimum size, minimum dataset size, ratio between conv/FC/RNN parameters?
  - what are some good example models/datasets to try?
- what are necessary steps to achieve scalable data-parallelism in Keras?
