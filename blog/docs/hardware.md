## Hardware

- connection of GPUs to CPU
    - PCIe
    - at best 16x lane,
    - multiple GPUs
- interconnection of GPUs
    - PCIe
        - on commodity hardware
    - NVLink
        - GPUs connected in a hyper-cube topology
        - on high-end hardware such as NVIDIA DGX-1
        - higher performance

[List of common GPUs for deep learning and their parameters](https://docs.google.com/spreadsheets/d/1pQNWTLfsBmclB3lojc5DHPZCRoKATjNqDA1Sbwq4zeE/edit#gid=0).

### Comparison of a custom 7-GPU machine with cloud instances

- `7gforce` - DIY machine - 6x 1070 + 1x 1080
    - too many devices share limited number of PCIe lanes
    - too slow host-device connection even for one device
    - in practice it's a bit limited for multi-GPU training
    - TODO: motherboard model (X99), CPU, cooling
- `az-2x-m60` - Azure Standard_NV12 (2x Tesla M60)
    - seems to work well
- `az-4x-m60` - Azure Standard_NV24 (4x Tesla M60) - TODO
- HW with NVLink (like NVIDIA DGX-1) would be much better
    - we don't have access to any -> can't measure

### Topology

Tip from https://github.com/BVLC/caffe/blob/master/docs/multigpu.md

```
$ nvidia-smi topo -m
```

```text
Legend:

  X   = Self
  SOC  = Connection traversing PCIe as well as the SMP link between CPU sockets(e.g. QPI)
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe switches (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing a single PCIe switch
  NV#  = Connection traversing a bonded set of # NVLinks
```

7gforce - two groups (within group just PCIe switch, between via CPU):
```text
        GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    CPU Affinity
GPU0     X      PIX     PIX     PHB     PHB     PHB     PHB     0-19
GPU1    PIX      X      PIX     PHB     PHB     PHB     PHB     0-19
GPU2    PIX     PIX      X      PHB     PHB     PHB     PHB     0-19
GPU3    PHB     PHB     PHB      X      PIX     PIX     PIX     0-19
GPU4    PHB     PHB     PHB     PIX      X      PIX     PIX     0-19
GPU5    PHB     PHB     PHB     PIX     PIX      X      PIX     0-19
GPU6    PHB     PHB     PHB     PIX     PIX     PIX      X      0-19
```

Azure 2x M60: - two GPUs on each board, but it seems the the GPUs communicate via QPI:
```text
GPU0    GPU1    CPU Affinity
GPU0     X     SOC    0-11
GPU1    SOC     X     0-11
```

TODO: measurements
- bandwidth
    - CUDA samples benchmark, nvprof from a training run
- training experiments

```
cd /usr/local/cuda/samples/1_Utilities/bandwidthTest
sudo make
./bandwidthTest
```

Measurements:

https://docs.google.com/spreadsheets/d/1c5yGydEANMzHjBufTzph0w-WGwJyiwPMRYz3yBZatb4/edit#gid=126374473

Observations:

- host <-> device bandwidth for the 1070 GPUs in the 7gforce machine is terribly low (8x lower than for others)
- host <-> device bandwidth for the 1080 GPU in 7gforce and M60 GPUs in Azure machine is comparable
- device to device bandwidth in the 1080 GPU is much high than for others
- device to device bandwidth in 1070 GPUs vary wildly (possibly depending on runtime configuration???)
