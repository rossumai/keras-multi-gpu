# More notes...

## Useful profiling tools

- nvprof
    - `nvprof python train.py`
    - `nvprof -o profiling.sqlite python train.py`
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
    - needs libcupti for GPU
        - a bit tricky to install on Ubuntu - we need to manually install a package from newer distribution version
    - possible/necessary to cut a small window while importing to the GUI
    - would be nice to make a tool to analyze/preprocess the SQLite outside the GUI
- TensorFlow profiler
    - https://medium.com/towards-data-science/howto-profile-tensorflow-1a49fb18073d
    - profiling at the level of TF operations
    - also needs libcupti
    - JSON output can be opened in chrome://tracing
    - records only latest session.run()
    - now possible to use with Keras but gives empty results
