#!/bin/bash

### Snatch YouTube video and extract into frames

START_TIME=0
LENGTH=0
VIDEO_NAME="clip"

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
        --url)
            echo "Downloading video..."
            youtube-dl -q -f best -o '%(input)s.%(ext)s' $2
            shift 2
            ;;
        # Variables
        -n|--name)
            VIDEO_NAME="$2"
            shift 2
            ;;
        -s|--start)
            START_TIME="$2"
            shift 2
            ;;
        -l|--length)
            LENGTH="$2"
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

if [ -z $OUTPUT_DIR ]; then
    OUTPUT_DIR="./youtube"
fi

if [ ! -d "$OUTPUT_DIR/" ]; then
    mkdir -p $OUTPUT_DIR
else
    rm -f $OUTPUT_DIR/*
fi

mv ./NA.mp4 ./$VIDEO_NAME.mp4

if [[ -z $LENGTH ]]; then
    ffmpeg -y -hide_banner -loglevel panic \
        -i ./$VIDEO_NAME.mp4 $OUTPUT_DIR/$VIDEO_NAME%05d.png \
        -q:a 0 -map a $VIDEO_NAME.mp3
else
    ffmpeg -y -hide_banner -loglevel panic \
        -ss $START_TIME \
        -t $LENGTH \
        -i ./$VIDEO_NAME.mp4 $OUTPUT_DIR/$VIDEO_NAME%05d.png \
        -q:a 0 -map a $VIDEO_NAME.mp3
fi
