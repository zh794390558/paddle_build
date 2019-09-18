#!/bin/bash

set -xe


export WORK_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")" && pwd )"
. ${WORK_ROOT}/image.sh
BUILD_ROOT=${WORK_ROOT}/build_docker${DOCKER_SUFFIX}

#export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

export FLAGS_fraction_of_gpu_memory_to_use=0.1
#export GLOG_v=4
unset CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=1
export FLAGS_benchmark=1

cd $BUILD_ROOT
#UNIT_TEST_NAME=test_concat_int8_mkldnn_op
#UNIT_TEST_NAME=test_concat_mkldnn_op
#UNIT_TEST_NAME=test_eager_deletion_recurrent_op
UNIT_TEST_NAME=test_fusion_group_op
#UNIT_TEST_NAME=device_code_test 
make test ARGS="-R ${UNIT_TEST_NAME} -V"

