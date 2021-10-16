#!/usr/bin/env bash

docker run --gpus all -it --rm \
    --shm-size=1g --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -p 8501:8501 \
	-v `pwd`:/scratch -w /scratch/ \
	stylegan:latest streamlit run StyleGAN/app.py
