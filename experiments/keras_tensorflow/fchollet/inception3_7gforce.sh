CUDA_VISIBLE_DEVICES=0           python benchmark_inception3_resnet50.py -a inception3 -m fchollet -p cpu
CUDA_VISIBLE_DEVICES=0,1         python benchmark_inception3_resnet50.py -a inception3 -m fchollet -p cpu
CUDA_VISIBLE_DEVICES=0,1,2       python benchmark_inception3_resnet50.py -a inception3 -m fchollet -p cpu
CUDA_VISIBLE_DEVICES=0,1,2,3     python benchmark_inception3_resnet50.py -a inception3 -m fchollet -p cpu
CUDA_VISIBLE_DEVICES=0,1,2,3,4   python benchmark_inception3_resnet50.py -a inception3 -m fchollet -p cpu
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python benchmark_inception3_resnet50.py -a inception3 -m fchollet -p cpu
