# time ./run.sh 2>&1 |tee -a 7gforce.txt
set -x

CUDA_VISIBLE_DEVICES=4   python kuza55_blog_example_cli.py -g 1 -p default
CUDA_VISIBLE_DEVICES=4,5 python kuza55_blog_example_cli.py -g 2 -p default
CUDA_VISIBLE_DEVICES=4,5 python kuza55_blog_example_cli.py -g 2 -p cpu
CUDA_VISIBLE_DEVICES=4,5 python kuza55_blog_example_cli.py -g 2 -p gpu
