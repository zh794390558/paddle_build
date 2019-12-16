#!/bin/bash

set -xe

PROJ_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")" && pwd )"
source ${PROJ_ROOT}/env.sh

export LD_LIBRARY_PATH=${THIRD_PARTY_PATH}/install/mklml/lib:${LD_LIBRARY_PATH}

#cd $BUILD_ROOT
#UNIT_TEST_NAME=test_step_rnn_lite_x86
#make test ARGS="-R ${UNIT_TEST_NAME} -V"

MODEL_DIR=${THIRD_PARTY_PATH}/install/step_rnn
${BUILD_ROOT}/lite/api/test_step_rnn_lite_x86 \
    --model_dir=${MODEL_DIR} \
    --repeats=100 \
    --threads=1 \
    --warmup=1
