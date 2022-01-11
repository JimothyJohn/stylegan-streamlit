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

# Fine-tune StyleGAN3-R for MetFaces-U using 1 GPU, starting from the pre-trained FFHQ-U pickle.
docker run --gpus all -it --rm \
    --shm-size=1g --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v $HOME/Datasets/Art:/data \
	-v `pwd`:/scratch -w /scratch/ \
	stylegan:latest python train.py --outdir=/scratch/training-runs/ --cfg=stylegan3-r --data=/scratch/art.zip \
    --gpus=1 --batch=4 --gamma=6.6 --mirror=1 --kimg=50 --snap=5
#    --resume=/scratch/models/stylegan3-r-ffhqu-256x256.pkl

curl -X POST \
    -H "Content-Type: application/json" \
    -d '{"username": "StyleGAN8GB", "content": "Training finished!"}' \
    $WEBHOOK_URL
