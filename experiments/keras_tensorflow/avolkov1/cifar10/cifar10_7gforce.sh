# ./cifar10_7gforce.sh 2>&1 | tee cifar10_7gforce_results.txt
# grep '^Samples/sec:' cifar10_7gforce_results.txt |sed 's#Samples/sec: ##' > cifar10_7gforce_times.csv
set -x

CMD="python ../../../../keras_tf_multigpu/examples/avolkov1/cifar/cifar10_cnn_mgpu.py"
EPOCHS="--epochs 10"
for BATCH_SIZE in 4096 512 32; do
  BATCH="--batch-size $BATCH_SIZE"
  RUN="$CMD $EPOCHS $BATCH"
  # baseline for 1 GPU - average
  CUDA_VISIBLE_DEVICES=0 $RUN
  CUDA_VISIBLE_DEVICES=1 $RUN
  CUDA_VISIBLE_DEVICES=4 $RUN
  CUDA_VISIBLE_DEVICES=5 $RUN
  for NCCL in "--nccl" ""; do
    # 2 GPUs - average
    CUDA_VISIBLE_DEVICES=0,1 $RUN --mgpu $NCCL
    CUDA_VISIBLE_DEVICES=4,5 $RUN --mgpu $NCCL
    # 3 GPUs - average
    CUDA_VISIBLE_DEVICES=0,1,2 $RUN --mgpu $NCCL
    CUDA_VISIBLE_DEVICES=4,5,6 $RUN --mgpu $NCCL
    # 4 GPUs
    CUDA_VISIBLE_DEVICES=0,1,2,3 $RUN --mgpu $NCCL
    # 5 GPUs
    CUDA_VISIBLE_DEVICES=0,1,2,3,4 $RUN --mgpu $NCCL
    # 6 GPUs
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 $RUN --mgpu $NCCL
  done
done
