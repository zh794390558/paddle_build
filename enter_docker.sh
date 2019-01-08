#!/bin/bash

set -xe

source image.sh

nvidia-docker run --name paddle_xreki_enter --network=host -it --rm -v $PWD/../../Paddle:/paddle -v $PWD/../../inference:/data -w /paddle \
    $XREKI_IMAGE_NAME:$XREKI_IMAGE_TAG \
    bash
