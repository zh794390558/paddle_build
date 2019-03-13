#!/bin/sh

set -xe

export XREKI_IMAGE_NAME=manylinux_trt
export XREKI_IMAGE_TAG=cuda8_cudnnv7
##export DOCKER_SUFFIX=_manylinux_cuda8

#export XREKI_IMAGE_NAME=xreki/paddle
#export XREKI_IMAGE_TAG=cuda92_cudnn7_debug
#export DOCKER_SUFFIX=_dev_cuda92

#export XREKI_IMAGE_NAME=hub.baidubce.com/paddlepaddle/paddle
#export XREKI_IMAGE_TAG=latest-dev
#export DOCKER_SUFFIX=_dev_cuda8
