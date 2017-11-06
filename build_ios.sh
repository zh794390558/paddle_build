#!/bin/bash

set -xe

PROJ_ROOT=$PWD

ARCH=arm64
if [ $# -ge 1 ]; then
    ARCH=$1
fi

SUFFIX=_ios_$ARCH

SOURCES_ROOT=$PROJ_ROOT/..

source common.sh

cd $BUILD_ROOT
if [ $ARCH == arm64 ]; then
    cmake -DCMAKE_INSTALL_PREFIX=$DEST_ROOT \
          -DTHIRD_PARTY_PATH=$THIRD_PARTY_PATH \
          -DCMAKE_SYSTEM_NAME=iOS \
          -DIOS_PLATFORM=OS \
          -DCMAKE_OSX_ARCHITECTURES="arm64" \
          -DIOS_USE_VECLIB_FOR_BLAS=ON \
          -DWITH_C_API=ON \
          -DWITH_TESTING=OFF \
          -DWITH_SWIG_PY=OFF \
          -DWITH_STYLE_CHECK=OFF \
          -DCMAKE_BUILD_TYPE=Release \
          $SOURCES_ROOT
elif [ $ARCH == armv7 ]; then
    cmake -DCMAKE_INSTALL_PREFIX=$DEST_ROOT \
          -DTHIRD_PARTY_PATH=$THIRD_PARTY_PATH \
          -DCMAKE_SYSTEM_NAME=iOS \
          -DIOS_PLATFORM=OS \
          -DCMAKE_OSX_ARCHITECTURES="armv7" \
          -DIOS_USE_VECLIB_FOR_BLAS=ON \
          -DWITH_C_API=ON \
          -DWITH_TESTING=OFF \
          -DWITH_SWIG_PY=OFF \
          -DWITH_STYLE_CHECK=OFF \
          -DCMAKE_BUILD_TYPE=Release \
          $SOURCES_ROOT
fi

cd $PROJ_ROOT
