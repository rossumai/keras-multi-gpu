# time ./benchmark_inception3_resnet50_7gforce.sh 2>&1 |tee -a benchmark_inception3_resnet50_results_7gforce.txt
set -x

for ARCH in inception3 resnet50; do
  for METHOD in kuza55 avolkov1; do
    RUN="python ../../keras_tf_multigpu/examples/benchmark_inception3_resnet50.py -a $ARCH -m $METHOD"
    CUDA_VISIBLE_DEVICES=0 $RUN -g 1
    CUDA_VISIBLE_DEVICES=0,1 $RUN -g 2 -p cpu
    CUDA_VISIBLE_DEVICES=0,1 $RUN -g 2 -p gpu
    CUDA_VISIBLE_DEVICES=0,1,2,3 $RUN -g 4 -p cpu
    CUDA_VISIBLE_DEVICES=0,1,2,3 $RUN -g 4 -p gpu
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 $RUN -g 6 -p cpu
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 $RUN -g 6 -p gpu
  done
done
