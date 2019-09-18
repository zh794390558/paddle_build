#!/bin/bash

set -xe

#export XREKI_IMAGE_NAME=manylinux_trt
#export XREKI_IMAGE_TAG=cuda8_cudnnv7
###export DOCKER_SUFFIX=_manylinux_cuda8

#export XREKI_IMAGE_NAME=paddlepaddle/paddle
#export XREKI_IMAGE_TAG=latest-gpu-cuda9.0-cudnn7-dev
#export DOCKER_SUFFIX=_dev_cuda90

export XREKI_IMAGE_NAME=paddlepaddle/paddle_manylinux_devel
export XREKI_IMAGE_TAG=cuda9.0_cudnn7
export DOCKER_SUFFIX=_manylinux_cuda90

function set_python_env() {
  export use_manylinux=`echo "${XREKI_IMAGE_NAME}" | grep manylinux | wc -l`
  if [ "${use_manylinux}" != "0" ];
  then
    export PYTHON_ABI="cp27-cp27mu"
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
}
