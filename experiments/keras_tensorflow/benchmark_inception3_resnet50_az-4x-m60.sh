# time ./benchmark_inception3_resnet50_az-4x-m60.sh 2>&1 |tee -a benchmark_inception3_resnet50_results_az-4x-m60.txt
set -x

pip freeze |grep Keras
pip freeze |grep tensorflow

for ARCH in inception3 resnet50; do
  for METHOD in fchollet avolkov1 kuza55; do
    RUN="python ../../keras_tf_multigpu/examples/benchmark_inception3_resnet50.py -a $ARCH -m $METHOD"
    CUDA_VISIBLE_DEVICES=0       $RUN -g 1
    CUDA_VISIBLE_DEVICES=0,1     $RUN -g 2 -p cpu
    CUDA_VISIBLE_DEVICES=0,1     $RUN -g 2 -p gpu
    CUDA_VISIBLE_DEVICES=0,1,2   $RUN -g 3 -p cpu
    CUDA_VISIBLE_DEVICES=0,1,2   $RUN -g 3 -p gpu
    CUDA_VISIBLE_DEVICES=0,1,2,3 $RUN -g 4 -p cpu
    CUDA_VISIBLE_DEVICES=0,1,2,3 $RUN -g 4 -p gpu
  done
done
