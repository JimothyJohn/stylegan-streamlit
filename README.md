# StyleGAN-Scripts

## Description

This is a Streamlit app for [StyleGAN3](https://github.com/NVlabs/stylegan3/). You'll want to go there for information since this is just a demo utility.

## Prereqs

- Debian-based Linux
- NVIDIA GPU >=8GB
- [CUDA >=11](https://developer.nvidia.com/cuda-downloads?target_os=Linux)
- [cuDNN >=8.2](https://developer.nvidia.com/cudnn-download-survey)

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

# Model Zoo 

|   Subject   |   Base  |   Resolution (px x px)  |   kimgs   |   FID |   Link
|-|-|-|-|-|-|
| Whataburger  |   StyleGAN3-R  |   256 |   1k                  | ???   | https://drive.google.com/file/d/1y6rNQr9lnYNtshQpeL0KvjkDStYwssIc/view?usp=sharing
| Monet |   StyleGAN3-R  |  256 |   1k                  | ???   | https://drive.google.com/file/d/1IKxpWjcuz0BWSaswyeXdUcC3ZCQxzCWM/view?usp=sharing
| Galaxy |   StyleGAN3-R  | 512 |   1k                  | 70   | https://drive.google.com/file/d/15QsTVf5CQhQqjWqq7_EWjFVP1Lgvi3FN/view?usp=sharing
| WikiArt |   StyleGAN3-R  | 1024 |   ???   | 8.1   | https://drive.google.com/file/d/18MOpwTMJsl_Z17q-wQVnaRLCUFZYSNkj/view?usp=sharing

# To-do

- Unify synthesis into single page (mappings)
- Add progress bars to UI
- Add video preview for mixing
- Accelerate projection algorithm using either previous vectors OR extract generator encoder
- Make device agnostic (.cuda() -> .to(device))
- Automate dataset creation wiht fiftyone
