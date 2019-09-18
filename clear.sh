#!/bin/bash

#### Clear the build directory
#if [ -d $BUILD_ROOT ]; then
#  rm -rf $BUILD_ROOT
#fi
mkdir -p -v $BUILD_ROOT

#### Clear the dist directory
if [ -d $DIST_ROOT ]; then
  rm -rf $DIST_ROOT
fi
