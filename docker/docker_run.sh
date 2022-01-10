#!/usr/bin/env bash

# Run the gen_images.py script using Docker:
docker run --gpus all -it --rm \
    --shm-size=1g --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v $HOME/Datasets/Art:/data \
    -v `pwd`:/scratch -w /scratch -e HOME=/scratch \
    stylegan:latest

#    python gen_images.py --outdir=out --trunc=1 --seeds=2 \
#    --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl
