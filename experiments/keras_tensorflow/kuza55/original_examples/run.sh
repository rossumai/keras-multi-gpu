# time ./run.sh 2>&1 |tee -a az-2x-m60.txt
set -x

CUDA_VISIBLE_DEVICES=0   python kuza55_blog_example_cli.py -g 1 -p default
CUDA_VISIBLE_DEVICES=0,1 python kuza55_blog_example_cli.py -g 2 -p default
CUDA_VISIBLE_DEVICES=0,1 python kuza55_blog_example_cli.py -g 2 -p cpu
CUDA_VISIBLE_DEVICES=0,1 python kuza55_blog_example_cli.py -g 2 -p gpu
