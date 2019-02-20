#!/bin/bash

set -xe

WITH_GPU=ON

PROJ_ROOT=/paddle/build_paddle
if [ $WITH_GPU == OFF ]; then
  SUFFIX=_docker_cpu
else
  SUFFIX=_docker
fi

SOURCES_ROOT=/paddle

function cmake_gen() {
  export CC=gcc
  export CXX=g++
  source $PROJ_ROOT/clear.sh
  cd $BUILD_ROOT
  cmake -DCMAKE_INSTALL_PREFIX=$DEST_ROOT \
        -DTHIRD_PARTY_PATH=$THIRD_PARTY_PATH \
        -DFLUID_INSTALL_DIR=$DEST_ROOT \
        -DCMAKE_BUILD_TYPE=Release \
        -DWITH_PROFILER=ON \
        -DGperftools_ROOT_DIR=/usr/local/lib \
        -DON_INFER=ON \
        -DWITH_DSO=ON \
        -DWITH_GPU=${WITH_GPU} \
        -DWITH_AMD_GPU=OFF \
        -DWITH_DISTRIBUTE=OFF \
        -DWITH_MKL=ON \
        -DWITH_NGRAPH=OFF \
        -DWITH_AVX=ON \
        -DCUDA_ARCH_NAME=Auto \
        -DWITH_PYTHON=ON \
        -DCUDNN_ROOT=/usr/ \
        -DWITH_TESTING=ON \
        -DCMAKE_MODULE_PATH=/opt/rocm/hip/cmake \
        -DON_INFER=ON \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DWITH_CONTRIB=ON \
        -DWITH_INFERENCE_API_TEST=ON \
        -DWITH_ANAKIN=ON \
        -DANAKIN_BUILD_FAT_BIN= \
        -DANAKIN_BUILD_CROSS_PLANTFORM= \
        -DPY_VERSION=2.7 \
        -DWITH_JEMALLOC=OFF \
        $SOURCES_ROOT
  cd $PROJ_ROOT
}

function build() {
  cd $BUILD_ROOT
  cat <<EOF
  ============================================
  Building in $BUILD_ROOT
  ============================================
EOF
  make -j12
  cd $PROJ_ROOT
}

function main() {
  local CMD=$1
  source $PROJ_ROOT/env.sh
  git config --global http.sslverify false
  case $CMD in
    cmake)
      cmake_gen
      ;;
    build)
      build
      ;;
    run)
      sh $PROJ_ROOT/run_docker.sh
#      cd $BUILD_ROOT
#      sh $PROJ_ROOT/run_test.sh
      ;;
  esac
}

main $@
