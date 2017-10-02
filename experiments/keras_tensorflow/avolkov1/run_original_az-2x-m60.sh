EPOCHS=5

# cd avolkov1-keras-experiments/
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python examples/cifar/cifar10_cnn_mgpu.py --epochs $EPOCHS
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1 python examples/cifar/cifar10_cnn_mgpu.py --mgpu --epochs $EPOCHS
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1 python examples/cifar/cifar10_cnn_mgpu.py --mgpu --epochs $EPOCHS --nccl

# python remove_progressbar_from_logs.py
# grep -A 1 '50000/50000' original_7gforce_results_nocr.txt |grep -A 2 'Epoch 5/5'|grep '50000/50000'
