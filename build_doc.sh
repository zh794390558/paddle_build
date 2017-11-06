#!/bin/bash

set -xe

PROJ_ROOT=$PWD
SUFFIX=_doc

SOURCES_ROOT=$PROJ_ROOT/..

source common.sh

# Compile Documentation only.
cd $BUILD_ROOT
cmake -DTHIRD_PARTY_PATH=$THIRD_PARTY_PATH \
      -DCMAKE_BUILD_TYPE=Debug \
      -DWITH_GPU=OFF \
      -DWITH_MKLDNN=OFF \
      -DWITH_MKLML=OFF \
      -DWITH_DOC=ON \
      $SOURCES_ROOT
make -j 12 gen_proto_py
make -j 12 paddle_docs paddle_docs_cn
