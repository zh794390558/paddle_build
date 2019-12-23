#!/bin/bash

set -xe

PROF_TOOL="none"
if [ $# -ge 1 ]; then
  PROF_TOOL=$1
  if [ ${PROF_TOOL} != "pprof" || ${PROF_TOOL} != "vtune" ]; then
    echo "Set profile tool to pprof or vtune"
    exit
  fi
fi

PROJ_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")" && pwd )"
source ${PROJ_ROOT}/env.sh

export LD_LIBRARY_PATH=${THIRD_PARTY_PATH}/install/mklml/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${GPERFTOOLS_ROOT}/lib:${LD_LIBRARY_PATH}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

#cd $BUILD_ROOT
#UNIT_TEST_NAME=test_step_rnn_lite_x86
#make test ARGS="-R ${UNIT_TEST_NAME} -V"

MODEL_DIR=${THIRD_PARTY_PATH}/install/step_rnn
if [ ${PROF_TOOL} == "none" ]; then
  ${BUILD_ROOT}/lite/api/test_step_rnn_lite_x86 \
      --model_dir=${MODEL_DIR} \
      --repeats=10000 \
      --threads=1 \
      --warmup=10
elif [ ${PROF_TOOL} == "pprof" ]; then
  ${GPERFTOOLS_ROOT}/bin/pprof --text ${BUILD_ROOT}/lite/api/test_step_rnn_lite_x86 step_rnn.x86.prof
elif [ ${PROF_TOOL} == "vtune" ]; then
  rm -rf r*hs
  rm -rf r*hs.tar.gz
  amplxe-cl -collect hotspots -knob sampling-mode=hw -knob enable-stack-collection=true -knob sampling-interval=1 \
      ${BUILD_ROOT}/lite/api/test_step_rnn_lite_x86 \
      --model_dir=${MODEL_DIR} \
      --repeats=100000 \
      --threads=1 \
      --warmup=1
fi
