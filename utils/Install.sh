#!/usr/bin/env bash

./docker/docker_build.sh
echo "export PATH=$(pwd)/utils:\$PATH" >> $HOME/.bashrc && . $HOME/.bashrc
