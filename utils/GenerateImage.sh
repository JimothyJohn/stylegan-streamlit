#!/usr/bin/env bash

# Default args
TRUNC=1
SEED="0-5"

PARAMS=""
POSITIONAL=()
while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        # Flags
        -h|--help)
            echo "Required inputs: --input, and --output"
            exit 1
            ;;  
        # Variables
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -n|--net)
            NET="$2"
            shift 2
            ;;
        -s|--seed)
            SEED="$2"
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

if [ ! -d "/input/${OUTPUT_DIR}" ]; then
    mkdir -p /input/${OUTPUT_DIR}
fi

docker run --gpus all -it --rm --shm-size=8g \
	-v `pwd`:/input \
    -v $HOME/github/stylegan-tools:/stylegan \
	-w /stylegan -e DNNLIB_CACHE_DIR=/stylegan/.cache \
    --user $(id -u):$(id -g) \
	stylegan:latest python /stylegan/generate.py \
	--trunc=${TRUNC} --seeds=${SEED} \
   	--outdir=/input/${OUTPUT_DIR} \
	--network=/stylegan/pretrained/${NET}.pkl
