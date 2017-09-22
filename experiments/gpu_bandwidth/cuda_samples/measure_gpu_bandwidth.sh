## Roughly measures GPU memory bandwidth for host-device and peer-to-peer
#
# Usage:
# ./measure_gpu_bandwidth.sh
#
# Output is written to bandwidthTest_gpu_$i.txt and p2pBandwidthLatencyTest.txt
#
# ----- bandwidthTest -----
#
# Example outout:
#
# [CUDA Bandwidth Test] - Starting...
# Running on...
#
#  Device 0: Tesla M60
#  Quick Mode
#
#  Host to Device Bandwidth, 1 Device(s)
#  PINNED Memory Transfers
#    Transfer Size (Bytes)	Bandwidth(MB/s)
#    33554432			6070.1
#
#  Device to Host Bandwidth, 1 Device(s)
#  PINNED Memory Transfers
#    Transfer Size (Bytes)	Bandwidth(MB/s)
#    33554432			6536.0
#
#  Device to Device Bandwidth, 1 Device(s)
#  PINNED Memory Transfers
#    Transfer Size (Bytes)	Bandwidth(MB/s)
#    33554432			133094.5
#
# Result = PASS
#
# NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.

set -x

CUDA_SAMPLES=~/cuda_samples
GPU_COUNT=$(nvidia-smi -L | wc -l)
for i in $(seq 0 $(expr $GPU_COUNT - 1)); do
  CUDA_VISIBLE_DEVICES=$i ${CUDA_SAMPLES}/1_Utilities/bandwidthTest/bandwidthTest |tee bandwidthTest_gpu_$i.txt
done

# ----- p2pBandwidthLatencyTest -----
#
# Example output:
# Unidirectional P2P=Disabled Bandwidth Matrix (GB/s)
#    D\D     0      1
#      0 114.34   5.26
#      1   5.26 116.87
# Unidirectional P2P=Enabled Bandwidth Matrix (GB/s)
#    D\D     0      1
#      0 120.24   5.26
#      1   5.27 134.50
# Bidirectional P2P=Disabled Bandwidth Matrix (GB/s)
#    D\D     0      1
#      0 120.76   5.47
#      1   5.46 135.50
# Bidirectional P2P=Enabled Bandwidth Matrix (GB/s)
#    D\D     0      1
#      0 135.42   5.48
#      1   5.47 135.92

CUDA_VISIBLE_DEVICES=$(seq -s, 0 $(expr $GPU_COUNT - 1)) ${CUDA_SAMPLES}/1_Utilities/p2pBandwidthLatencyTest/p2pBandwidthLatencyTest | tee p2pBandwidthLatencyTest.txt
