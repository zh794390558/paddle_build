#!/bin/bash

PROJ_ROOT=$PWD
SUFFIX=_android

SOURCES_ROOT=$PROJ_ROOT/..

source common.sh

ABI=32

if [ $# -eq 1 ]; then
    ABI=$1
fi

unset MKL_ROOT
unset OPENBLAS_ROOT

cd $BUILD_ROOT
if [ $ABI -eq 32 ]; then
    cmake -DCMAKE_INSTALL_PREFIX=$DEST_ROOT \
          -DTHIRD_PARTY_PATH=$THIRD_PARTY_PATH/armeabi-v7a \
          -DCMAKE_SYSTEM_NAME=Android \
          -DANDROID_STANDALONE_TOOLCHAIN=$ANDROID_ARM_TOOLCHAIN_ROOT \
          -DANDROID_ABI=armeabi-v7a \
          -DANDROID_ARM_NEON=ON \
          -DANDROID_ARM_MODE=ON \
          -DCMAKE_BUILD_TYPE=Release \
          -DWITH_C_API=ON \
          -DWITH_SWIG_PY=OFF \
          -DWITH_GOLANG=OFF \
          -DWITH_STYLE_CHECK=OFF \
          -DCMAKE_PREFIX_PATH="$JUMBO_ROOT" \
          $SOURCES_ROOT
elif [ $ABI -eq 64 ]; then
    cmake -DCMAKE_INSTALL_PREFIX=$DEST_ROOT \
          -DTHIRD_PARTY_PATH=${THIRD_PARTY_PATH}/arm64-v8a \
          -DCMAKE_SYSTEM_NAME=Android \
          -DANDROID_STANDALONE_TOOLCHAIN=$ANDROID_ARM64_TOOLCHAIN_ROOT \
          -DANDROID_ABI=arm64-v8a \
          -DANDROID_ARM_MODE=ON \
          -DCMAKE_BUILD_TYPE=Release \
          -DWITH_C_API=ON \
          -DWITH_SWIG_PY=OFF \
          -DWITH_GOLANG=OFF \
          -DWITH_STYLE_CHECK=OFF \
          -DCMAKE_PREFIX_PATH="$JUMBO_ROOT" \
          $SOURCES_ROOT
else
    echo "Invalid ABI ($ABI), you can set it to 32 or 64."
fi

cd $PROJ_ROOT
