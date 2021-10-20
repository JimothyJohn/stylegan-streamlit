# StyleGAN-Scripts

## Description

This is a Streamlit app for [StyleGAN3](https://github.com/NVlabs/stylegan3/). You'll want to go there for information since this is just a demo utility.

### Prerequisites

* Ubuntu (tested on 20.04)
* NVIDIA GPU
* [Docker](https://docs.docker.com/engine/install/ubuntu/#install-using-the-convenience-script)
* [NVIDIA Container Runtime](https://docs.docker.com/config/containers/resource_constraints/#gpuhttps://ngc.nvidia.com/signin)

## Setup

Build the docker image and run with:

```bash
user@host:~$ git clone https://github.com/JimothyJohn/stylegan-scripts # clone repo
user@host:~$ cd stylegan-scripts # move to repo directory
user@host:~$ docker/docker_build.sh # Build Docker image
user@host:~$ utils/Streamlit.sh # Run Streamlit app
```

## Functionality

You can run:

* Generation: Quickly generate images and save their mappings

* Projection: Extract mappings from your own images (in progress)

* Synthesis: Generate and combine images from mappings
