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

C_API=OFF

function cmake_gen() {
  export CC=gcc
  export CXX=g++
  source $PROJ_ROOT/clear.sh
  cd $BUILD_ROOT
  if [ $C_API == OFF ]; then
    cmake -DCMAKE_INSTALL_PREFIX=$DEST_ROOT \
          -DTHIRD_PARTY_PATH=$THIRD_PARTY_PATH \
          -DFLUID_INSTALL_DIR=$DEST_ROOT \
          -DCMAKE_BUILD_TYPE=RelWithDebInfo \
          -DWITH_PROFILER=ON \
          -DGperftools_ROOT_DIR=/usr/local/lib \
          -DON_INFER=ON \
          -DWITH_DSO=ON \
          -DWITH_DOC=OFF \
          -DWITH_GPU=${WITH_GPU} \
          -DWITH_AMD_GPU=OFF \
          -DWITH_DISTRIBUTE=OFF \
          -DWITH_MKL=OFF \
          -DWITH_AVX=ON \
          -DWITH_GOLANG=OFF \
          -DCUDA_ARCH_NAME=Auto \
          -DWITH_SWIG_PY=ON \
          -DWITH_C_API=OFF \
          -DWITH_PYTHON=ON \
          -DCUDNN_ROOT=/usr/ \
          -DWITH_TESTING=ON \
          -DCMAKE_MODULE_PATH=/opt/rocm/hip/cmake \
          -DWITH_FLUID_ONLY=ON \
          -DON_INFER=ON \
          -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
          -DWITH_CONTRIB=ON \
          -DWITH_INFERENCE_API_TEST=ON \
          -DWITH_ANAKIN=OFF \
          -DPY_VERSION=2.7 \
          $SOURCES_ROOT
  else
    cmake -DCMAKE_INSTALL_PREFIX=$DEST_ROOT \
          -DTHIRD_PARTY_PATH=$THIRD_PARTY_PATH \
          -DCUDA_ARCH_NAME=Auto \
          -DCMAKE_BUILD_TYPE=Release \
          -DWITH_C_API=ON \
          -DWITH_TIMER=ON \
          -DUSE_EIGEN_FOR_BLAS=OFF \
          -DWITH_PYTHON=OFF \
          -DWITH_MKL=OFF \
          -DWITH_GPU=ON \
          -DCUDNN_ROOT=$CUDNN_ROOT \
          -DWITH_SWIG_PY=OFF \
          -DWITH_GOLANG=OFF \
          $SOURCES_ROOT
  fi
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
