#!/bin/bash

set -xe

. /paddle/build_paddle/image.sh
BUILD_ROOT=/paddle/build_paddle/build_docker${DOCKER_SUFFIX}

export FLAGS_fraction_of_gpu_memory_to_use=0.1
#export GLOG_v=4
unset CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=1

cd $BUILD_ROOT
#UNIT_TEST_NAME=test_concat_int8_mkldnn_op
#UNIT_TEST_NAME=test_concat_mkldnn_op
#UNIT_TEST_NAME=concat_test
UNIT_TEST_NAME=test_rnn_encoder_decoder 
make test ARGS="-R ${UNIT_TEST_NAME} -V"

