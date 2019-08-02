#!/bin/bash

set -xe

. /paddle/build_paddle/image.sh
BUILD_ROOT=/paddle/build_paddle/build_docker${DOCKER_SUFFIX}

#export FLAGS_fraction_of_gpu_memory_to_use=0.1
#export GLOG_v=4
unset CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=1

CONFIG_ROOT=/paddle/paddle/fluid/operators/benchmark/configs
#FILENAME=${CONFIG_ROOT}/elementwise_add.config
#FILENAME=${CONFIG_ROOT}/gather.config
#FILENAME=${CONFIG_ROOT}/sequence_expand.config
#FILENAME=${CONFIG_ROOT}/is_empty.config
#FILENAME=${CONFIG_ROOT}/matmul.config
FILENAME=${CONFIG_ROOT}/fused_elemwise_activation.config
$BUILD_ROOT/paddle/fluid/operators/benchmark/op_tester \
    --op_config_list=${FILENAME} \
    --specified_config_id=0
