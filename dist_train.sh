#!/usr/bin/env bash

GPUS=$1
CONFIG=$2
PORT=${PORT:-4323}      # defualt: 4321

# usage
if [ $# -lt 2 ] ;then
    echo "usage:"
    echo "dist_train.sh [number of gpu] [path to option file]"
    exit
fi

# PYTHONPATH="$(dirname $0)/..:${PYTHONPATH}" \
# python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     lbasicsr/train.py -opt $CONFIG --launcher pytorch ${@:3}

# debug
# torchrun --nproc_per_node=$GPUS --master_port=$PORT lbasicsr/train.py -opt $CONFIG --debug --launcher pytorch

torchrun --nproc_per_node=$GPUS --master_port=$PORT lbasicsr/train.py -opt $CONFIG --launcher pytorch
