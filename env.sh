#!/bin/bash

NVCC=`which nvcc`
if [ ${NVCC} != "" ]; then
  NVCC_VERSION=`nvcc --version | tail -n 2 | grep "V[0-9][0-9]*\.[0-9]" -o | uniq`
  NVCC_VERSION=${NVCC_VERSION//V/}
  SUFFIX=${SUFFIX}"_cuda${NVCC_VERSION}"
fi
echo "NVCC_VERSION: ${NVCC_VERSION}"

GCC_VERSION=`gcc --version | head -n 1 | grep "[0-9]\.[0-9]\.[0-9]" -o | uniq`
SUFFIX=${SUFFIX}"_gcc${GCC_VERSION}"
echo "GCC_VERSION: ${GCC_VERSION}"

PY_VERSION=3.7.0
if [ "${PY_VERSION}" != "" ]; then
  SUFFIX=${SUFFIX}"_py${PY_VERSION}"
fi

BUILD_ROOT=${PROJ_ROOT}/build$SUFFIX
DEST_ROOT=$PROJ_ROOT/dist$SUFFIX
THIRD_PARTY_PATH=$PROJ_ROOT/third_party$SUFFIX

#### Set default sources root
if [ -z $SOURCES_ROOT ]; then
  SOURCES_ROOT=$PROJ_ROOT/..
fi

echo "PROJ_ROOT: ${PROJ_ROOT}"
echo "SOURCES_ROOT: ${SOURCES_ROOT}"
echo "BUILD_ROOT: ${BUILD_ROOT}"
echo "DEST_ROOT: ${DEST_ROOT}"
echo "THIRD_PARTY_PATH: ${THIRD_PARTY_PATH}"

OSNAME=`cat /etc/issue | head -n 1 | awk '{print $1}'`
echo "OS Name: $OSNAME"

function set_python_env() {
  if [ "${OSNAME}" == "CentOS" ];
  then
    if [ "${PY_VERSION}" == "3.5" ]; then
      export PYTHON_ABI="cp35-cp35m"
    elif  [ "${PY_VERSION}" == "3.6" ]; then
      export PYTHON_ABI="cp36-cp36m"
    elif  [ "${PY_VERSION}" == "3.7" ]; then
      export PYTHON_ABI="cp37-cp37m"
    else
      export PYTHON_ABI="cp27-cp27mu"
    fi

    echo "using python abi: $1"
    export LD_LIBRARY_PATH=/opt/python/${PYTHON_ABI}/lib:${LD_LIBRARY_PATH}
    export PATH=/opt/python/${PYTHON_ABI}/bin/:${PATH}
  else
    pyver=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
    if [ "$PY_VERSION" != "$pyver" ]; then
      echo "python version is not $PY_VERSION: $pyver"
      echo "source python${PY_VERSION} venv first."
      exit -1
    fi
    echo "python version: $pyver"
  fi
}
