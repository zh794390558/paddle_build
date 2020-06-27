#!/bin/bash

set -xe

WITH_GPU=ON
WITH_TESTING=ON

PROJ_ROOT=/paddle/build_paddle

SOURCES_ROOT=/paddle

function parse_version() {
  PADDLE_GITHUB_REPO=https://github.com/PaddlePaddle/Paddle.git
  git ls-remote --tags --refs ${PADDLE_GITHUB_REPO} > all_tags.txt
  latest_tag=`sed 's/refs\/tags\/v//g' all_tags.txt | awk 'END { print $NF }'`
  export PADDLE_VERSION=${latest_tag}
}

function cmake_gen() {
  export CC=gcc
  export CXX=g++
  source $PROJ_ROOT/clear.sh
#  export TENSORRT_ROOT=${PROJ_ROOT}/TensorRT-4.0.0.3
  cd $BUILD_ROOT
  if [ ${OSNAME} != "CentOS" ];
  then
    cmake -DCMAKE_INSTALL_PREFIX=$DEST_ROOT \
          -DTHIRD_PARTY_PATH=$THIRD_PARTY_PATH \
          -DFLUID_INSTALL_DIR=$DEST_ROOT \
          -DCMAKE_BUILD_TYPE=Release \
          -DWITH_PROFILER=OFF \
          -DGperftools_ROOT_DIR=/usr/local/lib \
          -DON_INFER=OFF \
          -DWITH_DSO=ON \
          -DWITH_GPU=${WITH_GPU} \
          -DWITH_AMD_GPU=OFF \
          -DWITH_DISTRIBUTE=OFF \
          -DWITH_DGC=OFF \
          -DWITH_MKL=ON \
          -DWITH_NGRAPH=OFF \
          -DWITH_AVX=ON \
          -DCUDA_ARCH_NAME=Auto \
          -DWITH_PYTHON=ON \
          -DCUDNN_ROOT=/usr/ \
          -DWITH_TESTING=ON \
          -DCMAKE_MODULE_PATH=/opt/rocm/hip/cmake \
          -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
          -DWITH_CONTRIB=ON \
          -DWITH_CRYPTO=OFF \
          -DWITH_INFERENCE_API_TEST=ON \
          -DPY_VERSION=${PY_VERSION} \
          $SOURCES_ROOT
  else
    if [ "$PYTHON_ABI" == "cp27-cp27m" ]; then
      PYTHON_EXECUTABLE=/opt/python/${PYTHON_ABI}/bin/python
      PYTHON_INCLUDE_DIR=/opt/python/${PYTHON_ABI}/include/python2.7
      PYTHON_LIBRARIES=/opt/python/${PYTHON_ABI}/lib/libpython2.7.so
      pip uninstall -y protobuf
      pip install -r ${SOURCES_ROOT}/python/requirements.txt
    elif [ "$PYTHON_ABI" == "cp27-cp27mu" ]; then
      PYTHON_EXECUTABLE=/opt/python/cp27-cp27mu/bin/python
      PYTHON_INCLUDE_DIR=/opt/python/cp27-cp27mu/include/python2.7
      PYTHON_LIBRARIES=/opt/python/${PYTHON_ABI}/lib/libpython2.7.so
      pip uninstall -y protobuf
      pip install -r ${SOURCES_ROOT}/python/requirements.txt
    elif [ "$PYTHON_ABI" == "cp35-cp35m" ]; then
      PYTHON_EXECUTABLE=/opt/python/${PYTHON_ABI}/bin/python3
      PYTHON_INCLUDE_DIR=/opt/python/${PYTHON_ABI}/include/python3.5m
      PYTHON_LIBRARIES=/opt/python/${PYTHON_ABI}/lib/libpython3.so
      pip3.5 uninstall -y protobuf
      pip3.5 install -r ${SOURCES_ROOT}/python/requirements.txt
    elif [ "$PYTHON_ABI" == "cp36-cp36m" ]; then
      PYTHON_EXECUTABLE=/opt/python/${PYTHON_ABI}/bin/python3
      PYTHON_INCLUDE_DIR=/opt/python/${PYTHON_ABI}/include/python3.6m
      PYTHON_LIBRARIES=/opt/python/${PYTHON_ABI}/lib/libpython3.so
      pip3.6 uninstall -y protobuf
      pip3.6 install -r ${SOURCES_ROOT}/python/requirements.txt
    elif [ "$PYTHON_ABI" == "cp37-cp37m" ]; then
      PYTHON_EXECUTABLE=/opt/python/${PYTHON_ABI}/bin/python3.7
      PYTHON_INCLUDE_DIR=/opt/python/${PYTHON_ABI}/include/python3.7m
      PYTHON_LIBRARIES=/opt/python/${PYTHON_ABI}/lib/libpython3.so
      pip3.7 uninstall -y protobuf
      pip3.7 install -r ${SOURCES_ROOT}/python/requirements.txt
    fi

    cmake -DCMAKE_INSTALL_PREFIX=$DEST_ROOT \
          -DTHIRD_PARTY_PATH=$THIRD_PARTY_PATH \
          -DFLUID_INSTALL_DIR=$DEST_ROOT \
          -DCMAKE_BUILD_TYPE=Release \
          -DWITH_GPU=${WITH_GPU} \
          -DCUDA_ARCH_NAME=Auto \
          -DWITH_PYTHON=ON \
          -DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE} \
          -DPYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIR} \
          -DPYTHON_LIBRARIES=${PYTHON_LIBRARIES} \
          -DWITH_AVX=ON \
          -DWITH_TESTING=${WITH_TESTING} \
          -DWITH_INFERENCE_API_TEST=ON \
          -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
          -DWITH_MKL=ON \
          -DWITH_DISTRIBUTE=ON \
          -DWITH_DGC=OFF \
          -DWITH_CRYPTO=OFF \
          -DCMAKE_VERBOSE_MAKEFILE=OFF \
          -DPY_VERSION=${PY_VERSION} \
          $SOURCES_ROOT
  fi
  cd $PROJ_ROOT
}

function build() {
  cd $BUILD_ROOT
  cat <<EOF
  ============================================
  Building in $BUILD_ROOT
  ============================================
EOF
  make -j12
  cd $PROJ_ROOT
}

function inference_lib() {
  cd $BUILD_ROOT
  cat <<EOF
  ============================================
  Copy inference libraries to $DEST_ROOT
  ============================================
EOF
  make inference_lib_dist -j12
  cd ${PROJ_ROOT}
}

function main() {
  local CMD=$1
  source $PROJ_ROOT/env.sh
  git config --global http.sslverify false
  set_python_env
  case $CMD in
    cmake)
#      parse_version
      cmake_gen
      ;;
    build)
#      parse_version
      build
      ;;
    inference_lib)
      inference_lib
      ;;
    run)
      sh $PROJ_ROOT/run_docker.sh
#      cd $BUILD_ROOT
#      sh $PROJ_ROOT/run_test.sh
      ;;
    version)
      parse_version
      ;;
  esac
}

main $@
