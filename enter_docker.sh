#!/bin/bash

set -xe

source image.sh

nvidia-docker run --name build_paddle_lyq${DOCKER_SUFFIX} --network=host -it --rm \
    -v $PWD/../../Paddle:/paddle \
    -v $PWD/../../models:/models \
    -v $PWD/../../benchmark:/benchmark \
    -v $PWD/../../inference:/data \
    -w /paddle \
    $XREKI_IMAGE_NAME:$XREKI_IMAGE_TAG \
    bash

