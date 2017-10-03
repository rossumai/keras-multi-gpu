# time ./run_az-2x-m60.sh 2>&1 |tee -a az-2x-m60.txt
set -x

for ARCH in inception3 resnet50; do
  for METHOD in kuza55 avolkov1; do
    RUN="python ../../keras_tf_multigpu/examples/benchmark_inception3_resnet50.py -a $ARCH -m $METHOD"
    CUDA_VISIBLE_DEVICES=0   $RUN -g 1
    CUDA_VISIBLE_DEVICES=0,1 $RUN -g 2 -p cpu
    CUDA_VISIBLE_DEVICES=0,1 $RUN -g 2 -p gpu
  done
done
