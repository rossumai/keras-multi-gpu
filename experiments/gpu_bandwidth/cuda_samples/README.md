# Measurement of GPU bandwidth (host<->device, device<->device)

Needs CUDA Toolkit installed.

```
# Copy CUDA samples locally and compile the required commands
./gpu_bandwidth_init.sh

# Measure bandwidth on each GPU and then P2P bandwidth between all GPU pairs
./measure_gpu_bandwidth.sh
```

NOTE: CUDA Samples state this is just example code, not a good benchmark. But we don't have anything better now and the measurements are probably not far from truth.

The results will go into text files in the current directory.

- measured results: https://docs.google.com/spreadsheets/d/1c5yGydEANMzHjBufTzph0w-WGwJyiwPMRYz3yBZatb4/edit#gid=126374473
- [list of NVIDIA GPUs specs](https://docs.google.com/spreadsheets/d/1pQNWTLfsBmclB3lojc5DHPZCRoKATjNqDA1Sbwq4zeE/edit#gid=0) (with max. memory bandwidth)
