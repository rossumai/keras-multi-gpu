# Towards efficient multi-GPU training in Keras over TensorFlow

_A summary blog post for publishing on Medium and additional resources on GitHub (Markdown documents)._

The blog post message:

- currently multi-gpu training is already possible in Keras
    - various third-party scripts + now in upstream (upcoming 2.0.9)
    - it runs, it able to get some speed-up, but not as high as possible
- also there are some third-party packages (horovod, tensorpack)
    - little bit more complicated API/installation
    - claim good speed-up
- good speed-up with plain tensorflow is possible but complicated
    - this is the goal
- ideally we'd like good speed-up with simple API
    - existing Keras multi-gpu code with some imporovements:
        - double-buffering of batches at GPU using StagingArea
        - providing data to TF memory asynchronously - using TF queues or Dataset API
- more in-depth research on this topic:
  - algorithms and techniques
  - hardware consideration
  - other implementations
  - implementations in Keras + TensorFlow
  - our experiments and measurements
