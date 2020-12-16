#!/bin/bash

set -xe

export PROJ_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")" && pwd )"
. ${PROJ_ROOT}/env.sh

set_python_env

export FLAGS_fraction_of_gpu_memory_to_use=0.1
unset CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES="1"
#export FLAGS_benchmark=1

cd $BUILD_ROOT
UNIT_TEST_NAME=test_ir_fusion_group_pass
#UNIT_TEST_NAME=test_code_generator
#export GLOG_vmodule=fusion_group_pass=4
#export GLOG_vmodule=operator=4
#export GLOG_v=4
make test ARGS="-R ${UNIT_TEST_NAME} -V"

