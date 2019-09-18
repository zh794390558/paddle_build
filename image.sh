#!/bin/sh

set -xe

#export XREKI_IMAGE_NAME=manylinux_trt
#export XREKI_IMAGE_TAG=cuda8_cudnnv7
###export DOCKER_SUFFIX=_manylinux_cuda8

#export XREKI_IMAGE_NAME=xreki/paddle
#export XREKI_IMAGE_TAG=cuda92_cudnn7_debug
##export DOCKER_SUFFIX=_dev_cuda92

#export XREKI_IMAGE_NAME=paddlepaddle/paddle
#export XREKI_IMAGE_TAG=latest-gpu-cuda9.0-cudnn7-dev
#export DOCKER_SUFFIX=_dev_cuda90

export XREKI_IMAGE_NAME=paddlepaddle/paddle_manylinux_devel
export XREKI_IMAGE_TAG=cuda9.0_cudnn7
export DOCKER_SUFFIX=_manylinux_cuda90

