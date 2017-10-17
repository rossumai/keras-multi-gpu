mkdir data/
cp ~/.keras/datasets/cifar-10-batches-py/* data/
for file in data/*_batch*; do
  mv $file ${file}.bin
done

CUDA_VISIBLE_DEVICES=0 python cifar10_main.py \
  --data_dir=data/ \
  --model_dir=/tmp/cifar10 \
  --is_cpu_ps=True \
  --force_gpu_compatible=True \
  --num_gpus=1 \
  --train_steps=1000

CUDA_VISIBLE_DEVICES=0,1 python cifar10_main.py \
  --data_dir=data/ \
  --model_dir=/tmp/cifar10 \
  --is_cpu_ps=True \
  --force_gpu_compatible=True \
  --num_gpus=2 \
  --train_steps=1000
