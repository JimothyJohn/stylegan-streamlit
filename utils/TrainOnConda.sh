#!/usr/bin/env bash

PARAMS=""
DATASET_NAME="figurative-paintings.zip"
CONFIG="stylegan3-t"
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

python train.py \
    --outdir=$OUT_DIR \
    --cfg=$CONFIG --data=$DATASET_NAME \
    --gpus=1 --batch=4 --gamma=2 \
    --metrics=none --kimg=1000 --snap=20 \
#    --resume="models/network-snapshot-000560.pkl"
