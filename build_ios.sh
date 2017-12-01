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
if [ $ARCH == os -o $ARCH == arm64 -o $ARCH == armv7 -o $ARCH == armv7s ]; then
    if [ $ARCH == os ]; then
        IOS_ARCH="armv7;armv7s;arm64"
    else
        IOS_ARCH="$ARCH"
    fi
    cmake -DCMAKE_INSTALL_PREFIX=$DEST_ROOT \
          -DTHIRD_PARTY_PATH=$THIRD_PARTY_PATH \
          -DCMAKE_SYSTEM_NAME=iOS \
          -DIOS_PLATFORM=OS \
          -DIOS_ARCH=$IOS_ARCH \
          -DIOS_ENABLE_BITCODE=ON \
          -DIOS_USE_VECLIB_FOR_BLAS=ON \
          -DWITH_C_API=ON \
          -DWITH_TESTING=OFF \
          -DWITH_SWIG_PY=OFF \
          -DWITH_STYLE_CHECK=OFF \
          -DCMAKE_BUILD_TYPE=MinSizeRel \
          $SOURCES_ROOT
elif [ $ARCH == simulator -o $ARCH == x86_64 -o $ARCH == i386 ]; then
    if [ $ARCH == simulator ]; then
        IOS_ARCH="i386;x86_64"
    else
        IOS_ARCH="$ARCH"
    fi
    cmake -DCMAKE_INSTALL_PREFIX=$DEST_ROOT \
          -DTHIRD_PARTY_PATH=$THIRD_PARTY_PATH \
          -DCMAKE_SYSTEM_NAME=iOS \
          -DIOS_PLATFORM=SIMULATOR \
          -DIOS_ARCH=$IOS_ARCH \
          -DIOS_ENABLE_BITCODE=ON \
          -DIOS_USE_VECLIB_FOR_BLAS=ON \
          -DWITH_C_API=ON \
          -DWITH_TESTING=OFF \
          -DWITH_SWIG_PY=OFF \
          -DWITH_STYLE_CHECK=OFF \
          -DCMAKE_BUILD_TYPE=MinSizeRel \
          $SOURCES_ROOT
fi

cd $PROJ_ROOT
