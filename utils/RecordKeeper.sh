#!/usr/bin/env bash

OUTPUT_DIR="./data"

PARAMS=""
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -h|--help)
            echo "Required args: --input, --output, --seed, --net"
            exit 1
            ;;  
        -i|--input)
            INPUT_DIR="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -*|--*=) # unsupported flags
            echo "Error: Unsupported flag $1" >&2
            shift
            ;;
        *) # preserve positional arguments
            PARAMS="$PARAMS $1"
            ;;
    esac
done

# set positional arguments in their proper place
eval set -- "$PARAMS"

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p $OUTPUT_DIR
fi

if [ ! -d "$INPUT_DIR" ]; then
    mkdir -p $INPUT_DIR
fi

docker run --gpus all -it --rm --shm-size=8g \
	-v `pwd`:/input -w /stylegan \
    -v $HOME/github/stylegan-tools:/stylegan \
	-e DNNLIB_CACHE_DIR=/stylegan/.cache \
	stylegan:latest python dataset_tool.py \
    --source=$INPUT_DIR \
    --dest=$OUTPUT_DIR.zip
