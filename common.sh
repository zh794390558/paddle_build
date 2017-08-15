#!/bin/bash

BUILD_ROOT=${PROJ_ROOT}/build$SUFFIX
DEST_ROOT=$PROJ_ROOT/dist$SUFFIX
THIRD_PARTY_PATH=$PROJ_ROOT/third_party$SUFFIX

#### Set default sources root
if [ -z $SOURCES_ROOT ]; then
    SOURCES_ROOT=$PROJ_ROOT/..
fi

echo $PROJ_ROOT
echo $SOURCES_ROOT
echo $THIRD_PARTY_PATH

#### Clear the build directory
if [ -d $BUILD_ROOT ]; then
    rm -rf $BUILD_ROOT
fi
mkdir -p -v $BUILD_ROOT

