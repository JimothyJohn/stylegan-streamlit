#!/usr/bin/env bash

SEED="0"

PARAMS=""
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        # Flags
        -h|--help)
            echo "Required args: --input, --output, --style, --net"
            exit 1
            ;;  
        # Variables
        -i|--input)
            IMAGE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -s|--style)
            STYLE="$2"
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

if [ ! -d "/input/${OUTPUT_FILE}" ]; then
    mkdir -p /input/${OUTPUT_FILE}
fi

docker run --gpus all -it --rm --shm-size=8g \
	-v `pwd`:/input -w /stylegan \
    -v $HOME/github/stylegan-tools:/stylegan \
	-e DNNLIB_CACHE_DIR=/stylegan/.cache \
	stylegan:pytorch python /stylegan/vectorvideo.py \
	--output=/input/$OUTPUT_FILE --seed=${SEED} --target=${IMAGE} \
    --network=/stylegan/pretrained/${NET}.pkl $SAVE
