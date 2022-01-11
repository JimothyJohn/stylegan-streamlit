#!/usr/bin/env bash

PARAMS=""
DATASET_NAME="art.zip"
CONFIG="stylegan3-r"
OUT_DIR="training-runs/"
case "$1" in
    -d|--dataset)
        DATASET_NAME="$1"
        ;;
    -*|--*=) # unsupported flags
        echo "Error: Unsupported flag $1" >&2
        exit 1
        ;;
    *) # preserve positional arguments
        PARAMS="$PARAMS $1"
        ;;
esac

# set positional arguments in their proper place
eval set -- "$PARAMS"

# Train StyleGAN3 from scratch on 4 x A100's
/opt/conda/envs/stylegan/bin/python train.py \
    --outdir=$OUT_DIR \
    --cfg=$CONFIG --data=$DATASET_NAME \
    --gpus=4 --batch=32 --gamma=6.6 --mirror=1 --kimg=100 --snap=5
#    --resume=/scratch/models/stylegan3-r-ffhqu-256x256.pkl
