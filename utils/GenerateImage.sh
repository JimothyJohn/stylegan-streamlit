#!/usr/bin/env bash

# Default args
TRUNC=1
SEED="0-5"
# NET="models/stylegan3-r-ffhqu-256x256.pkl"
NET="models/stylegan3-r-ffhq-1024x1024.pkl"
OUTPUT_DIR="out/"

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

if [ ! -d "${OUTPUT_DIR}" ]; then
    mkdir -p ${OUTPUT_DIR}
fi

docker run --gpus all -it --rm \
    --shm-size=1g --ulimit memlock=-1 \
    --ulimit stack=67108864 \
	-v `pwd`:/scratch \
	-w /scratch \
	stylegan:latest python gen_images.py \
	--trunc=${TRUNC} --seeds=${SEED} \
   	--outdir=${OUTPUT_DIR} \
	--network=$NET
