#!/bin/bash

set -xe

PROJ_ROOT=/paddle/build_paddle
SUFFIX=_docker

SOURCES_ROOT=/paddle

unset MKL_ROOT
C_API=OFF
if [ $# -ge 1 ]; then
    if [ $1 -eq 1 ]; then
        C_API=ON
        SUFFIX=_capi
    fi
fi

function cmake_gen() {
  source common.sh

  cd $BUILD_ROOT
  if [ $C_API == OFF ]; then
    cmake -DCMAKE_INSTALL_PREFIX=$DEST_ROOT \
          -DTHIRD_PARTY_PATH=$THIRD_PARTY_PATH \
          -DFLUID_INSTALL_DIR=$DEST_ROOT \
          -DCMAKE_BUILD_TYPE=Release \
          -DWITH_DSO=ON \
          -DWITH_DOC=OFF \
          -DWITH_GPU=ON \
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
          -DWITH_FAST_BUNDLE_TEST=ON \
          -DCMAKE_MODULE_PATH=/opt/rocm/hip/cmake \
          -DWITH_FLUID_ONLY=OFF \
          -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
          -DWITH_CONTRIB=ON \
          -DWITH_INFERENCE=ON \
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
  Building in $BUILD_ROOT ...
  ============================================
EOF
  make -j8
  cd $PROJ_ROOT
}

function main() {
  local CMD=$1
  case $CMD in
    cmake)
      cmake_gen
      ;;
    build)
      build
      ;;
  esac
}

main $@
