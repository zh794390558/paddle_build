#!/bin/bash

set -xe

PROJ_ROOT=$PWD
SUFFIX=

SOURCES_ROOT=$PROJ_ROOT/..

source common.sh

C_API=OFF
if [ $# -ge 1 ]; then
    if [ $1 -eq 1 ]; then
        C_API=ON
    fi
fi

export CC=gcc
export CXX=g++

cd $BUILD_ROOT
if [ $C_API == OFF ]; then
  cmake -DCMAKE_INSTALL_PREFIX=$DEST_ROOT \
        -DTHIRD_PARTY_PATH=$THIRD_PARTY_PATH \
        -DCMAKE_BUILD_TYPE=Release \
        -DWITH_PYTHON=ON \
        -DWITH_SWIG_PY=ON \
        -DWITH_GPU=OFF \
        -DWITH_MKLML=OFF \
        -DWITH_MKLDNN=OFF \
        -DWITH_GOLANG=OFF \
        -DWITH_STYLE_CHECK=OFF \
        $SOURCES_ROOT
else
  cmake -DCMAKE_INSTALL_PREFIX=$DEST_ROOT \
        -DTHIRD_PARTY_PATH=$THIRD_PARTY_PATH \
        -DCMAKE_BUILD_TYPE=Release \
        -DWITH_C_API=ON \
        -DWITH_PYTHON=OFF \
        -DWITH_SWIG_PY=OFF \
        -DWITH_GPU=OFF \
        -DWITH_MKLML=OFF \
        -DWITH_MKLDNN=OFF \
        -DWITH_GOLANG=OFF \
        -DWITH_STYLE_CHECK=OFF \
        $SOURCES_ROOT
fi

cd $PROJ_ROOT

