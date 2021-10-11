#!/usr/bin/env bash

# Run the gen_images.py script using Docker:
docker run --gpus all -it --rm \
    --shm-size=1g --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=unix${DISPLAY} \
    -e PYTHONPATH="/scratch/dnnlib:/scratch/torch_utils" \
    --user $(id -u):$(id -g) \
    -v `pwd`:/scratch -w /scratch -e HOME=/scratch \
    stylegan:latest

#    python gen_images.py --outdir=out --trunc=1 --seeds=2 \
#    --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl
