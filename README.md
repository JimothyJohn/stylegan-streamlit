# StyleGAN-Scripts

## Description

This is my toolset for [StyleGAN3](https://github.com/NVlabs/stylegan3/). You'll want to go there for information since this is mostly just a Docker script wrapper for the common Python utilities.

### Prerequisites

* Ubuntu (tested on 20.04)
* NVIDIA GPU
* [Docker](https://docs.docker.com/engine/install/ubuntu/#install-using-the-convenience-script)
* [NVIDIA Container Runtime](https://docs.docker.com/config/containers/resource_constraints/#gpuhttps://ngc.nvidia.com/signin)

## Setup

Build the docker image and add script commands with:

```bash
user@host:~$ git clone https://github.com/JimothyJohn/stylegan-scripts # clone repo
user@host:~$ cd stylegan-scripts # move to repo directory
user@host:~$ utils/Install.sh # Install docker and utilities
```

## Utilities

All utilities can be found under [utils/](./utils):

* [Download.sh](./utils/Download.sh) - Download tool for limited datasets and most pretrained models. (you will need shape-predictor)
* [AlignFaces.sh](./utils/AlignFaces.sh) - Single-instance human face alignment via Nvidia's [FFHQ method](https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py). (required before projections can be done)
* [ProjectImage.sh](./utils/ProjectImage.sh) - Projects a single input image into latent space (allows you to blend styles)
* [GenerateImage.sh](./utils/GenerateImage.sh) - Use the Generator to create a new image based on an output seed of the pretrained model. Not too useful, but a nice "Hello world"
