# time ./run.sh 2>&1 |tee az-2x-m60.txt
set -x

CUDA_VISIBLE_DEVICES=0 python kuza55_blog_example_cli.py -p default -g 1
CUDA_VISIBLE_DEVICES=0,1 python kuza55_blog_example_cli.py -p default -g 2
CUDA_VISIBLE_DEVICES=0 python kuza55_blog_example_cli.py -p cpu -g 1
CUDA_VISIBLE_DEVICES=0,1 python kuza55_blog_example_cli.py -p cpu -g 2
CUDA_VISIBLE_DEVICES=0 python kuza55_blog_example_cli.py -p gpu -g 1
CUDA_VISIBLE_DEVICES=0,1 python kuza55_blog_example_cli.py -p gpu -g 2
