#!/bin/bash

set -xe

source image.sh

#docker run --name paddle_trt -i --rm ${CUDA_SO} ${DEVICES} -v $PWD:/paddle -w /paddle \
#    -e "CMAKE_BUILD_TYPE=RelWithDebInfo" \
#    -e "WITH_GPU=ON" \
#    -e "WITH_MKL=OFF" \
#    -e "WITH_AVX=ON" \
#    -e "CUDA_ARCH_NAME=Auto" \
#    -e "WITH_C_API=OFF" \
#    -e "WITH_TESTING=ON" \
#    -e "WITH_FLUID_ONLY=ON" \
#    -e "WITH_DEB=OFF" \
#    -e "WITH_ANAKIN=OFF" \
#    -e "RUN_TEST=OFF" \
#    -e "INSTALL_PREFIX=/paddle/dist" \
#    paddlepaddle/paddle:latest-dev \
#    paddle/scripts/paddle_build.sh build

NAME=xreki/paddle
TAG=cuda92_cudnn7_debug
#NVIDIA_DOCKER_PATH=/home/liuyiqun/packages/nvidia-docker-2.0.3
nvidia-docker run --name paddle_xreki_enter --network=host -it --rm \
    -v $PWD/../../Paddle:/paddle \
    -v $PWD/../../models:/models \
    -v $PWD/../../inference:/data \
    -w /paddle \
    $XREKI_IMAGE_NAME:$XREKI_IMAGE_TAG \
    bash
