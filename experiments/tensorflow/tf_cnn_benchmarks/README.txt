```
git clone https://github.com/tensorflow/benchmarks tensorflow-benchmarks
export TF_BENCHMARKS_PATH=tensorflow-benchmarks/scripts/tf_cnn_benchmarks
export MACHINE=az2xm60
CUDA_VISIBLE_DEVICES=0,1 time ./benchmark_batch.sh $TF_BENCHMARKS_PATH 2>&1 |tee benchmark_batch_results_$MACHINE_$(date +%Y-%m-%d).txt
```

Suite run time: 13m

https://docs.google.com/spreadsheets/d/1c5yGydEANMzHjBufTzph0w-WGwJyiwPMRYz3yBZatb4/edit?usp=sharing
