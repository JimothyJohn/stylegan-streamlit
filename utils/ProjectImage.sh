#!/usr/bin/env bash

SEED="0"
dGENERATOR="models/tylegan3-r-ffhq-1024x1024.pkl"

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
    mkdir -p /input/${OUTPUT_DIR}/vectors/
fi

docker run --gpus all -it --rm \
    --shm-size=1g --ulimit memlock=-1 \
    --ulimit stack=67108864 \
	-v `pwd`:/scratch -w /scratch \
	stylegan:latest python /scratch/projector.py \
	--outdir=$OUTPUT_DIR --seed=$SEED --target=$INPUT_IMG \
    --network=$GENERATOR
