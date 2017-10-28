# $git clone https://github.com/tensorflow/benchmarks.git ~/tf-benchmarks
# master depends on TF 1.4 (allreduce), checkout older commit:
# $ git checkout 6a33b4a4b5bda950bb7e45faf13120115cbfdb2f
#
# run as:
# $ time ./benchmark_az-4x-m60.sh ~/tf-benchmarks/scripts/tf_cnn_benchmarks 2>&1 |tee benchmark_batch_results_az-4x-m60.txt
#
set -x

TF_BENCHMARKS_PATH=~/tf-benchmarks

for MODEL in inception3 resnet50; do
  RUN="python ${TF_BENCHMARKS_PATH}/tf_cnn_benchmarks.py --model $MODEL"
  CUDA_VISIBLE_DEVICES=0 $RUN --num_gpus 1
  CUDA_VISIBLE_DEVICES=0,1 $RUN --num_gpus 2 --local_parameter_device cpu
  CUDA_VISIBLE_DEVICES=0,1 $RUN --num_gpus 2 # PS=GPU
  CUDA_VISIBLE_DEVICES=0,1,2 $RUN --num_gpus 3 --local_parameter_device cpu
  CUDA_VISIBLE_DEVICES=0,1,2,3 $RUN --num_gpus 4 --local_parameter_device cpu
  CUDA_VISIBLE_DEVICES=0,1,2,3 $RUN --num_gpus 4 # PS=GPU
  CUDA_VISIBLE_DEVICES=0,1,2,3 $RUN --num_gpus 4 --variable_update replicated
  CUDA_VISIBLE_DEVICES=0,1,2,3 $RUN --num_gpus 4 --variable_update replicated --use_nccl False
  CUDA_VISIBLE_DEVICES=0,1,2,3 $RUN --num_gpus 4 --variable_update independent
done


for BATCH_SIZE in 128 512; do
  RUN="python ${TF_BENCHMARKS_PATH}/tf_cnn_benchmarks.py --model resnet56 --data_name cifar10 --data_dir $HOME/.keras/datasets/cifar-10-batches-py"
  CUDA_VISIBLE_DEVICES=0 $RUN --num_gpus 1
  CUDA_VISIBLE_DEVICES=0,1 $RUN --num_gpus 2 --local_parameter_device cpu
  CUDA_VISIBLE_DEVICES=0,1 $RUN --num_gpus 2 # PS=GPU
  CUDA_VISIBLE_DEVICES=0,1,2 $RUN --num_gpus 3 --local_parameter_device cpu
  CUDA_VISIBLE_DEVICES=0,1,2,3 $RUN --num_gpus 4 --local_parameter_device cpu
  CUDA_VISIBLE_DEVICES=0,1,2,3 $RUN --num_gpus 4 # PS=GPU
  CUDA_VISIBLE_DEVICES=0,1,2,3 $RUN --num_gpus 4 --variable_update replicated
  CUDA_VISIBLE_DEVICES=0,1,2,3 $RUN --num_gpus 4 --variable_update replicated --use_nccl False
  CUDA_VISIBLE_DEVICES=0,1,2,3 $RUN --num_gpus 4 --variable_update independent
done
