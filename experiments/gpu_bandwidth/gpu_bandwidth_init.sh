set -x

cp -r /usr/local/cuda/samples/ ~/cuda_samples

# http://docs.nvidia.com/cuda/cuda-samples/index.html#bandwidth-test
cd ~/cuda_samples/1_Utilities/bandwidthTest
make

# http://docs.nvidia.com/cuda/cuda-samples/index.html#peer-to-peer-bandwidth-latency-test-with-multi-gpus
cd ~/cuda_samples/1_Utilities/p2pBandwidthLatencyTest
make
