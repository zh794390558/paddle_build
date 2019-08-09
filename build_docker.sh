#!/bin/bash

set -xe
source image.sh
use_manylinux=`echo "${XREKI_IMAGE_NAME}" | grep manylinux | wc -l`

WITH_GPU=ON

PROJ_ROOT=/paddle/build_paddle
if [ $WITH_GPU == OFF ]; then
  SUFFIX=_docker_cpu${DOCKER_SUFFIX}
else
  SUFFIX=_docker${DOCKER_SUFFIX}
fi

SOURCES_ROOT=/paddle

if [ ${use_manylinux} != "0" ];
then
  PYTHON_ABI="cp27-cp27mu"
  echo "using python abi: $1"
  if [ "$PYTHON_ABI" == "cp27-cp27m" ]; then
    export LD_LIBRARY_PATH=/opt/_internal/cpython-2.7.11-ucs2/lib:${LD_LIBRARY_PATH#/opt/_internal/cpython-2.7.11-ucs4/lib:}
    export PATH=/opt/python/cp27-cp27m/bin/:${PATH}
  elif [ "$PYTHON_ABI" == "cp27-cp27mu" ]; then
    export LD_LIBRARY_PATH=/opt/_internal/cpython-2.7.11-ucs4/lib:${LD_LIBRARY_PATH#/opt/_internal/cpython-2.7.11-ucs2/lib:}
    export PATH=/opt/python/cp27-cp27mu/bin/:${PATH}
  elif [ "$PYTHON_ABI" == "cp35-cp35m" ]; then
    export LD_LIBRARY_PATH=/opt/_internal/cpython-3.5.1/lib/:${LD_LIBRARY_PATH}
    export PATH=/opt/_internal/cpython-3.5.1/bin/:${PATH}
  elif [ "$PYTHON_ABI" == "cp36-cp36m" ]; then
    export LD_LIBRARY_PATH=/opt/_internal/cpython-3.6.0/lib/:${LD_LIBRARY_PATH}
    export PATH=/opt/_internal/cpython-3.6.0/bin/:${PATH}
  elif [ "$PYTHON_ABI" == "cp37-cp37m" ]; then
    export LD_LIBRARY_PATH=/opt/_internal/cpython-3.7.0/lib/:${LD_LIBRARY_PATH}
    export PATH=/opt/_internal/cpython-3.7.0/bin/:${PATH}
  fi
fi

function cmake_gen() {
  export CC=gcc
  export CXX=g++
  source $PROJ_ROOT/clear.sh
  cd $BUILD_ROOT
  if [ ${use_manylinux} == "0" ];
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
          -DWITH_INFERENCE_API_TEST=ON \
          -DPY_VERSION=2.7 \
          $SOURCES_ROOT
  else
    if [ "$PYTHON_ABI" == "cp27-cp27m" ]; then
      export LD_LIBRARY_PATH=/opt/_internal/cpython-2.7.11-ucs2/lib:${LD_LIBRARY_PATH#/opt/_internal/cpython-2.7.11-ucs4/lib:}
      export PATH=/opt/python/cp27-cp27m/bin/:${PATH}
      PYTHON_EXECUTABLE=/opt/python/cp27-cp27m/bin/python
      PYTHON_INCLUDE_DIR=/opt/python/cp27-cp27m/include/python2.7
      PYTHON_LIBRARIES=/opt/_internal/cpython-2.7.11-ucs2/lib/libpython2.7.so
      pip uninstall -y protobuf
      pip install -r ${SOURCES_ROOT}/python/requirements.txt
    elif [ "$PYTHON_ABI" == "cp27-cp27mu" ]; then
      export LD_LIBRARY_PATH=/opt/_internal/cpython-2.7.11-ucs4/lib:${LD_LIBRARY_PATH#/opt/_internal/cpython-2.7.11-ucs2/lib:}
      export PATH=/opt/python/cp27-cp27mu/bin/:${PATH}
      PYTHON_EXECUTABLE=/opt/python/cp27-cp27mu/bin/python
      PYTHON_INCLUDE_DIR=/opt/python/cp27-cp27mu/include/python2.7
      PYTHON_LIBRARIES=/opt/_internal/cpython-2.7.11-ucs4/lib/libpython2.7.so
      pip uninstall -y protobuf
      pip install -r ${SOURCES_ROOT}/python/requirements.txt
    elif [ "$PYTHON_ABI" == "cp35-cp35m" ]; then
      export LD_LIBRARY_PATH=/opt/_internal/cpython-3.5.1/lib/:${LD_LIBRARY_PATH}
      export PATH=/opt/_internal/cpython-3.5.1/bin/:${PATH}
      PYTHON_EXECUTABLE=/opt/_internal/cpython-3.5.1/bin/python3
      PYTHON_INCLUDE_DIR=/opt/_internal/cpython-3.5.1/include/python3.5m
      PYTHON_LIBRARIES=/opt/_internal/cpython-3.5.1/lib/libpython3.so
      pip3.5 uninstall -y protobuf
      pip3.5 install -r ${SOURCES_ROOT}/python/requirements.txt
    elif [ "$PYTHON_ABI" == "cp36-cp36m" ]; then
      export LD_LIBRARY_PATH=/opt/_internal/cpython-3.6.0/lib/:${LD_LIBRARY_PATH}
      export PATH=/opt/_internal/cpython-3.6.0/bin/:${PATH}
      PYTHON_EXECUTABLE=/opt/_internal/cpython-3.6.0/bin/python3
      PYTHON_INCLUDE_DIR=/opt/_internal/cpython-3.6.0/include/python3.6m
      PYTHON_LIBRARIES=/opt/_internal/cpython-3.6.0/lib/libpython3.so
      pip3.6 uninstall -y protobuf
      pip3.6 install -r ${SOURCES_ROOT}/python/requirements.txt
    elif [ "$PYTHON_ABI" == "cp37-cp37m" ]; then
      export LD_LIBRARY_PATH=/opt/_internal/cpython-3.7.0/lib/:${LD_LIBRARY_PATH}
      export PATH=/opt/_internal/cpython-3.7.0/bin/:${PATH}
      PYTHON_EXECUTABLE=/opt/_internal/cpython-3.7.0/bin/python3.7
      PYTHON_INCLUDE_DIR=/opt/_internal/cpython-3.7.0/include/python3.7m
      PYTHON_LIBRARIES=/opt/_internal/cpython-3.7.0/lib/libpython3.so
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
          -DWITH_TESTING=ON \
          -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
          -DWITH_MKL=ON \
          -DWITH_DISTRIBUTE=OFF \
          -DWITH_DGC=OFF \
          -DCMAKE_VERBOSE_MAKEFILE=OFF \
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
#  make ocr_plate_tester -j12
#  make op_tester -j12
#  make concat_test -j12
  make -j12
  cd $PROJ_ROOT
}

function main() {
  local CMD=$1
  source $PROJ_ROOT/env.sh
  git config --global http.sslverify false
  case $CMD in
    cmake)
      cmake_gen
      ;;
    build)
      build
      ;;
    run)
      sh $PROJ_ROOT/run_docker.sh
#      cd $BUILD_ROOT
#      sh $PROJ_ROOT/run_test.sh
      ;;
  esac
}

main $@
