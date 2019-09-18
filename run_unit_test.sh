#!/bin/bash

set -xe
. /paddle/build_paddle/image.sh
set_python_env

SOURCES_ROOT=/paddle
BUILD_ROOT=/paddle/build_paddle/build_docker${DOCKER_SUFFIX}
export PYTHONPATH=/paddle/python

export FLAGS_fraction_of_gpu_memory_to_use=0.1
#export GLOG_v=4
unset CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=5

cd $BUILD_ROOT
#UNIT_TEST_NAME=test_concat_int8_mkldnn_op
UNIT_TEST_NAME=test_fused_fc_elementwise_layernorm_op
#UNIT_TEST_NAME=test_simplify_with_basic_ops_pass
#UNIT_TEST_NAME=test_fc_elementwise_layernorm_fuse_pass
make test ARGS="-R ${UNIT_TEST_NAME} -V"

