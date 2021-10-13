# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

FROM nvcr.io/nvidia/pytorch:21.08-py3

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV DEBIAN_FRONTEND=noninteractive
ENV LIBGL_ALWAYS_INDIRECT=1

RUN apt-get update
RUN apt-get -y install libglfw3-dev mesa-utils libgl1-mesa-glx
RUN pip install imageio-ffmpeg==0.4.4 pyspng==0.1.0 imgui glfw pyopengl imageio dlib

WORKDIR /workspace

# RUN (printf '#!/bin/bash\nexec \"$@\"\n' >> /entry.sh) && chmod a+x /entry.sh
# ENTRYPOINT ["/entry.sh"]
