#!/bin/bash

set -xe

PROJ_ROOT=$PWD
SUFFIX=

SOURCES_ROOT=$PROJ_ROOT/..

unset MKL_ROOT
C_API=OFF
if [ $# -ge 1 ]; then
    if [ $1 -eq 1 ]; then
        C_API=ON
        SUFFIX=_capi
    fi
fi

source env.sh
source clear.sh

cd $BUILD_ROOT
if [ $C_API == OFF ]; then
  cmake -DCMAKE_INSTALL_PREFIX=$DEST_ROOT \
        -DTHIRD_PARTY_PATH=$THIRD_PARTY_PATH \
        -DFLUID_INSTALL_DIR=$DEST_ROOT \
        -DCUDA_ARCH_NAME=Auto \
        -DCMAKE_BUILD_TYPE=Release \
        -DWITH_FLUID_ONLY=OFF \
        -DWITH_C_API=OFF \
        -DWITH_PYTHON=ON \
        -DWITH_MKL=OFF \
        -DWITH_DISTRIBUTE=OFF \
        -DWITH_GPU=ON \
        -DCUDNN_ROOT=$CUDNN_ROOT \
        -DWITH_SWIG_PY=ON \
        -DWITH_TESTING=ON \
        -DWITH_INFERENCE_API_TEST=ON \
        -DWITH_GOLANG=OFF \
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
        -DCMAKE_PREFIX_PATH="$JUMBO_ROOT" \
        $SOURCES_ROOT
fi

cd $PROJ_ROOT

