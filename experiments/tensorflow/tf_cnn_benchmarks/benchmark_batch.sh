# run as:
# time ./benchmark_batch.sh tensorflow-benchmarks/scripts/tf_cnn_benchmarks 2>&1 |tee benchmark_batch_results.txt
set -x

TF_BENCHMARKS_PATH=$1

# models: alexnet, densenet, googlenet, inception, lenet, overfeat, resnet, trivial, vgg
# models that don't need imagenet dataset: alexnet, googlenet, lenet, overfeat, trivial

# defaults:
# --local_parameter_device gpu
# --variable_update parameter_server
# --use_nccl True (only for --variable_update replicated)
# --data_format NCHW
# compare these to: python ${TF_BENCHMARKS_PATH}/tf_cnn_benchmarks.py --model alexnet --num_gpus 2
python ${TF_BENCHMARKS_PATH}/tf_cnn_benchmarks.py --model alexnet --num_gpus 2 --local_parameter_device cpu
python ${TF_BENCHMARKS_PATH}/tf_cnn_benchmarks.py --model alexnet --num_gpus 2 --variable_update replicated
python ${TF_BENCHMARKS_PATH}/tf_cnn_benchmarks.py --model alexnet --num_gpus 2 --variable_update replicated --use_nccl False
python ${TF_BENCHMARKS_PATH}/tf_cnn_benchmarks.py --model alexnet --num_gpus 2 --variable_update independent
python ${TF_BENCHMARKS_PATH}/tf_cnn_benchmarks.py --model alexnet --num_gpus 2 --data_format NHWC

for model in trivial alexnet lenet googlenet overfeat; do
  for num_gpus in 1 2; do
    python ${TF_BENCHMARKS_PATH}/tf_cnn_benchmarks.py --model $model --num_gpus $num_gpus
  done
done
