#!/bin/bash

set -xe


export WORK_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")" && pwd )"
. ${WORK_ROOT}/env.sh
BUILD_ROOT=${WORK_ROOT}/build_docker${DOCKER_SUFFIX}

set_python_env

export FLAGS_fraction_of_gpu_memory_to_use=0.1
#export GLOG_v=4
unset CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=1
export FLAGS_benchmark=1

cd $BUILD_ROOT
UNIT_TEST_NAME=test_fusion_group_pass
make test ARGS="-R ${UNIT_TEST_NAME} -V"

