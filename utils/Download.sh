#!/usr/bin/env bash

mkdir -p "./models"

helpmenu() { 
    echo "Required inputs: --net or --data"
    echo "Networks available: celeba-hq, afhq(dog,cat,wild), ffhq, shape-predictor"
    echo "Datasets available: celeba-hq, afhq"
    exit 1
}

PARAMS=""
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        # Flags
        -h|--help)
            helpmenu
            ;;
        # Variables
        -d|--data)
            DATA="$2"
            shift 2
            ;;
        -n|--net)
            NET="$2"
            shift 2
            ;;
        -*|--*=) # unsupported flags
            echo "Error: Unsupported flag $2" >&2
            shift
            ;;
        *) # preserve positional arguments
            PARAMS="$PARAMS $1"
            ;;
    esac
done

# set positional arguments in their proper place
eval set -- "$PARAMS"

if [ ! -z $NET ]; then
    case $NET in
        # Models
        "celeba-hq")
            URL=https://www.dropbox.com/s/96fmei6c93o8b8t/100000_nets_ema.ckpt?dl=0
            OUT_FILE=./celeba-hq-100000.ckpt
            wget -N $URL -O $OUT_FILE
            ;;
        "afhqcat")
            wget -O ./pretrained/afhqcat.pkl https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqcat.pkl
            ;;
        "afhqdog")
            wget -O ./pretrained/afhqcat.pkl https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqdog.pkl
            ;;
        "afhqwild")
            wget -O ./pretrained/afhqcat.pkl https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqwild.pkl
            ;;
        "metfaces")
            wget -O ./pretrained/afhqcat.pkl https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
            ;;
        "ffhq")
            wget -O ./pretrained/ffhq.pkl https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
            ;;
        "shape-predictor")
            wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
            bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
            mv shape_predictor_68_face_landmarks.dat ./pretrained/shape_predictor.dat
            ;;
        "vgg16")
            wget -O ./pretrained/vgg16.pt https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt
            ;;
        *) # unsupported flags
            echo "Error: Unsupported network \"$NET\"" >&2
            echo "Networks available: celeba-hq, afhq(dog,cat,wild), ffhq, shape-predictor"
            exit 1
            ;;
    esac
else
    case $DATA in
        # Models
        "celeba-hq")
            URL=https://www.dropbox.com/s/f7pvjij2xlpff59/celeba_hq.zip?dl=0
            ZIP_FILE=celeba-hq.zip
            wget -N $URL -O $ZIP_FILE
            unzip $ZIP_FILE -d ./celeba-hq
            rm $ZIP_FILE
            ;;
        "afhq")
            URL=https://www.dropbox.com/s/t9l9o3vsx2jai3z/afhq.zip?dl=0
            ZIP_FILE=./afhq.zip
            wget -N $URL -O $ZIP_FILE
            unzip $ZIP_FILE -d ./afhq
            rm $ZIP_FILE
            ;;
        *) # unsupported flags
            echo "Error: Unsupported dataset \"$DATA\"" >&2
            echo "Datasets available: celeba-hq, afhq"
            exit 1
            ;;
    esac
fi
