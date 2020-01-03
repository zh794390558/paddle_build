#!/bin/bash

set -xe

PROJ_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")" && pwd )"
SOURCES_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")/../" && pwd )"

source $PROJ_ROOT/env.sh

#cd ${SOURCES_ROOT}
#${SOURCES_ROOT}/lite/tools/ci_build.sh prepare_workspace

function cmake_gen() {
  source ${PROJ_ROOT}/clear.sh
  cd ${BUILD_ROOT}
  cmake -DTHIRD_PARTY_PATH=${THIRD_PARTY_PATH} \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DWITH_PYTHON=OFF \
        -DWITH_MKL=ON \
        -DWITH_MKLDNN=OFF \
        -DWITH_LITE=ON \
        -DWITH_TESTING=ON \
        -DLITE_WITH_CUDA=OFF \
        -DLITE_WITH_X86=ON \
        -DLITE_WITH_ARM=OFF \
        -DLITE_BUILD_EXTRA=ON \
        -DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=OFF \
        -DLITE_WITH_PROFILE=OFF \
        ${SOURCES_ROOT}
  cd ${PROJ_ROOT}
}

function build() {
  cd $BUILD_ROOT
  cat <<EOF
  ============================================
  Building in $BUILD_ROOT
  ============================================
EOF
#  make -j12
#  make test_step_rnn_lite_x86 VERBOSE=1
  make test_step_rnn_lite_x86 -j12
  cd $PROJ_ROOT
}

function main() {
  local CMD=$1
  git config --global http.sslverify false
  case $CMD in
    cmake)
      cmake_gen
      ;;
    build)
      build
      ;;
    run)
      sh $PROJ_ROOT/run_lite.sh
      ;;
  esac
}

main $@
