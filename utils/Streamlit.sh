#!/usr/bin/env bash

docker run --gpus all -d \
    --name stylegan \
    --shm-size=1g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -p 8501:8501 -e HOME=/scratch \
	-v `pwd`:/scratch -w /scratch/ \
	stylegan:latest streamlit run StyleGAN/app.py
