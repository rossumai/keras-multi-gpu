# Use CIFAR10 dataset in ~/.keras/datasets/cifar-10-batches-py
# Downloaded and extracted by keras:
#
# >>> from keras.datasets import cifar10
# >>> cifar10.load_data()
#
# time ./benchmark_batch_cifar10.sh tensorflow-benchmarks/scripts/tf_cnn_benchmarks 2>&1 |tee benchmark_batch_results_cifar10.txt
#
# Runtime around 50 min on az-2x-m60
set -x

TF_BENCHMARKS_PATH=$1

# densenet100_k24 with batch_size=64 failed on OOM
for MODEL in alexnet resnet20 resnet32 resnet44 resnet56 resnet110 densenet40_k12 densenet100_k12; do
  RUN="python ${TF_BENCHMARKS_PATH}/tf_cnn_benchmarks.py --model $MODEL --data_name cifar10 --data_dir $HOME/.keras/datasets/cifar-10-batches-py"
  CUDA_VISIBLE_DEVICES=0 $RUN --num_gpus 1
  CUDA_VISIBLE_DEVICES=0 $RUN --num_gpus 1 --data_format NHWC
  CUDA_VISIBLE_DEVICES=0,1 $RUN --num_gpus 2
  CUDA_VISIBLE_DEVICES=0,1 $RUN --num_gpus 2 --local_parameter_device cpu
  for USE_NCCL in true false; do
    CUDA_VISIBLE_DEVICES=0,1 $RUN --num_gpus 2 --variable_update replicated --use_nccl $USE_NCCL
  done
  CUDA_VISIBLE_DEVICES=0,1 $RUN --num_gpus 2 --variable_update independent
done
