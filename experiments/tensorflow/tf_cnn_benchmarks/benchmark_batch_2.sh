# run as:
# time ./benchmark_batch_2.sh tensorflow-benchmarks/scripts/tf_cnn_benchmarks 2>&1 |tee benchmark_batch_results_inception3.txt
set -x

TF_BENCHMARKS_PATH=$1

python ${TF_BENCHMARKS_PATH}/tf_cnn_benchmarks.py --model inception3 --num_gpus 1
python ${TF_BENCHMARKS_PATH}/tf_cnn_benchmarks.py --model inception3 --num_gpus 2
python ${TF_BENCHMARKS_PATH}/tf_cnn_benchmarks.py --model inception3 --num_gpus 2 --local_parameter_device cpu
python ${TF_BENCHMARKS_PATH}/tf_cnn_benchmarks.py --model inception3 --num_gpus 2 --variable_update replicated
python ${TF_BENCHMARKS_PATH}/tf_cnn_benchmarks.py --model inception3 --num_gpus 2 --variable_update replicated --use_nccl False
python ${TF_BENCHMARKS_PATH}/tf_cnn_benchmarks.py --model inception3 --num_gpus 2 --variable_update independent

python ${TF_BENCHMARKS_PATH}/tf_cnn_benchmarks.py --model resnet50_v2 --num_gpus 1
python ${TF_BENCHMARKS_PATH}/tf_cnn_benchmarks.py --model resnet50_v2 --num_gpus 2
python ${TF_BENCHMARKS_PATH}/tf_cnn_benchmarks.py --model resnet50_v2 --num_gpus 2 --local_parameter_device cpu
python ${TF_BENCHMARKS_PATH}/tf_cnn_benchmarks.py --model resnet50_v2 --num_gpus 2 --variable_update replicated
python ${TF_BENCHMARKS_PATH}/tf_cnn_benchmarks.py --model resnet50_v2 --num_gpus 2 --variable_update replicated --use_nccl False
python ${TF_BENCHMARKS_PATH}/tf_cnn_benchmarks.py --model resnet50_v2 --num_gpus 2 --variable_update independent
