#!/usr/bin/env bash

SEED="3"
# NET="models/stylegan3-r-ffhqu-256x256.pkl"
NET="models/stylegan3-r-ffhq-1024x1024.pkl"
OUTPUT_DIR="out/"
STEPS="500"

PARAMS=""
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        # Flags
        -h|--help)
            echo "Required args: --input, --output, --seed, --net"
            exit 1
            ;;  
        # Variables
        -i|--input)
            INPUT_IMG="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -n|--net)
            NET="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
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

if [[ $INPUT_IMG = "" ]]; then
    echo "Please choose an input image with --input"
    exit 1
fi

docker run --gpus all -it --rm \
    --shm-size=1g --ulimit memlock=-1 \
    --ulimit stack=67108864 \
	-v `pwd`:/scratch -w /scratch \
	stylegan:latest python projector.py \
	--outdir=$OUTPUT_DIR --seed=$SEED \
    --target=$INPUT_IMG --network=$NET \
    --save-video="True" --num-steps=$STEPS
