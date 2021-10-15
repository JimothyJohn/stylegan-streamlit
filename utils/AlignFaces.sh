#!/usr/bin/env bash

# Default values
PREDICTOR="models/shape-predictor.dat"

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
        -i|--input)
            INPUT_DIR="$2"
            shift 2
            ;;
        # Variables
        -v|--video)
            VIDEO_FILE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --json)
            JSON="$2"
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

if [[ "$INPUT_DIR" = "" ]]; then
    echo "Please choose an input directory/video with \"-i\""
    exit 1
fi

if [[ "$OUTPUT_DIR" = "" ]]; then
    echo "Please choose an output directory with \"-o\""
    exit 1
fi

if [ -d "$OUTPUT_DIR/" ]; then
    rm -r $OUTPUT_DIR
fi

mkdir -p $OUTPUT_DIR/json

if [[ -z "$JSON" ]]; then
    JSON="--json=$OUTPUT_DIR/json"
fi

# If we're processing a video
if [[ ! -z $VIDEO_FILE ]]; then
    # Clear temporary frames folder
    rm ./frames/*
    # Slice video into frames and save audio
    ffmpeg -y -hide_banner -loglevel panic \
        -i ${VIDEO_FILE} ./frames/%05d.png \
        -q:a 0 -map a ./clip.mp3
    # Set align target location
    $INPUT_IMG=./frames
fi

docker run --gpus all -it --rm \
    --shm-size=1g --ulimit memlock=-1 \
    --ulimit stack=67108864 \
	-v `pwd`:/scratch -w /scratch \
	stylegan:latest python alignFaces.py \
    --predictor=$PREDICTOR \
    --input=$INPUT_DIR \
    --output=$OUTPUT_DIR $JSON
