# StyleGAN-Scripts

## Description

This is a Streamlit app for [StyleGAN3](https://github.com/NVlabs/stylegan3/). You'll want to go there for information since this is just a demo utility.

## Prereqs

* Debian-based Linux
* NVIDIA GPU >=8GB
* [CUDA >=11](https://developer.nvidia.com/cuda-downloads?target_os=Linux)
* [cuDNN >=8.2](https://developer.nvidia.com/cudnn-download-survey)

## Setup

Clone the repo and move on in:

```bash
sudo apt update && sudo apt -y install git
git clone https://github.com/JimothyJohn/stylegan-scripts # clone repo
cd stylegan-scripts # move to repo directory
```

### [Anaconda](https://docs.conda.io/en/latest/miniconda.html#linux-installers)

Install Anaconda environment and initialize with:

```bash
conda env create -f environment.yml
conda activate stylegan 
```

### [Docker](https://docs.docker.com/engine/install/ubuntu/)

Build Docker image and run with:

```bash
docker/docker_build.sh
utils/Streamlit.sh
```