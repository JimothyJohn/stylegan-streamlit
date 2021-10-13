#!/usr/bin/env bash

PARAMS=""
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

curl -X POST \
    -H "Content-Type: application/json" \
    -d '{"username": "StyleGAN8GB", "content": "Training started!"}' \
    $WEBHOOK_URL

python train.py \
    --outdir=$DATASET_ROOT/training-runs \
    --data=$DATASET_ROOT/$DATASET_NAME.zip \
    --gpus=8

curl -X POST \
    -H "Content-Type: application/json" \
    -d '{"username": "StyleGAN8GB", "content": "Training finished!"}' \
    $WEBHOOK_URL
