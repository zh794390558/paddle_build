#!/bin/bash

set -xe

PROJ_ROOT=$PWD
SUFFIX=_android

SOURCES_ROOT=$PROJ_ROOT/..

source common.sh

ABI=0
if [ $# -eq 1 ]; then
    ABI=$1
fi

unset MKL_ROOT
unset OPENBLAS_ROOT

cd $BUILD_ROOT
if [ $ABI -eq 32 ]; then
    ANDROID_ABI=armeabi-v7a
    cmake -DCMAKE_INSTALL_PREFIX=$DEST_ROOT \
          -DTHIRD_PARTY_PATH=$THIRD_PARTY_PATH/$ANDROID_ABI \
          -DCMAKE_SYSTEM_NAME=Android \
          -DANDROID_STANDALONE_TOOLCHAIN=$ANDROID_ARM_STANDALONE_TOOLCHAIN \
          -DANDROID_ABI=$ANDROID_ABI \
          -DANDROID_ARM_NEON=ON \
          -DANDROID_ARM_MODE=ON \
          -DCMAKE_BUILD_TYPE=Release \
          -DWITH_C_API=ON \
          -DUSE_EIGEN_FOR_BLAS=ON \
          -DWITH_SWIG_PY=OFF \
          -DWITH_GOLANG=OFF \
          -DWITH_STYLE_CHECK=OFF \
          -DCMAKE_PREFIX_PATH="$JUMBO_ROOT" \
          $SOURCES_ROOT
elif [ $ABI -eq 64 ]; then
    ANDROID_ABI=arm64-v8a
    cmake -DCMAKE_INSTALL_PREFIX=$DEST_ROOT \
          -DTHIRD_PARTY_PATH=${THIRD_PARTY_PATH}/$ANDROID_ABI \
          -DCMAKE_SYSTEM_NAME=Android \
          -DANDROID_STANDALONE_TOOLCHAIN=$ANDROID_ARM64_STANDALONE_TOOLCHAIN \
          -DANDROID_ABI=$ANDROID_ABI \
          -DANDROID_ARM_MODE=ON \
          -DCMAKE_BUILD_TYPE=Release \
          -DWITH_C_API=ON \
          -DWITH_SWIG_PY=OFF \
          -DWITH_GOLANG=OFF \
          -DWITH_STYLE_CHECK=OFF \
          -DCMAKE_PREFIX_PATH="$JUMBO_ROOT" \
          $SOURCES_ROOT
else
    echo "Build for armeabi default."
    ANDROID_ABI=armeabi
    cmake -DCMAKE_INSTALL_PREFIX=$DEST_ROOT \
          -DTHIRD_PARTY_PATH=$THIRD_PARTY_PATH/$ANDROID_ABI \
          -DCMAKE_SYSTEM_NAME=Android \
          -DANDROID_STANDALONE_TOOLCHAIN=$ANDROID_ARM_STANDALONE_TOOLCHAIN \
          -DANDROID_ABI=$ANDROID_ABI \
          -DANDROID_ARM_MODE=ON \
          -DCMAKE_BUILD_TYPE=Release \
          -DWITH_C_API=ON \
          -DWITH_SWIG_PY=OFF \
          -DWITH_GOLANG=OFF \
          -DWITH_STYLE_CHECK=OFF \
          -DCMAKE_PREFIX_PATH="$JUMBO_ROOT" \
          $SOURCES_ROOT
fi

cd $PROJ_ROOT
