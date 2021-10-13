#!/usr/bin/env bash

#!/usr/bin/env bash

ROWS="0-3"
COLS="4-7"

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
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -n|--net)
            NET="$2"
            shift 2
            ;;
		-r|--rows)
            ROW="$2"
            shift 2
            ;;
		-c|--cols)
            COL="$2"
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
else
    rm /input/${OUTPUT_DIR}/*
fi

docker run --gpus all -it --rm --shm-size=8g \
	-v `pwd`:/input -w /stylegan \
	-v $HOME/github/stylegan-tools:/stylegan \
	-e DNNLIB_CACHE_DIR=/stylegan/.cache \
	stylegan:latest python /stylegan/style_mixing.py \
    --outdir=/input/$OUTPUT_DIR \
	--network=/stylegan/pretrained/${NET}.pkl \
	--rows=${ROWS} --cols=${COLS}
