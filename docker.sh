#!/bin/sh

set -xe

TASK=run
if [ $# -ge 1 ]; then
  if [ $1 -eq 1 ]; then
    TASK=cmake
  elif [ $1 -eq 2 ]; then
    TASK=build
  fi
fi

# docker build -t xreki/paddle:cuda92_cudnn7_debug .
# docker login
# username: xreki
# password:
# docker push xreki/paddle:cuda92_cudnn7_debug

#CUDA_ROOT=/usr
#CUDA_ROOT=/usr/local/cuda
#export CUDA_SO="$(\ls ${CUDA_ROOT}/lib64/libcuda* | xargs -I{} echo '-v {}:{}') $(\ls ${CUDA_ROOT}/lib64/libnvidia* | xargs -I{} echo '-v {}:{}')"
#export DEVICES=$(\ls /dev/nvidia* | xargs -I{} echo '--device {}:{}')

#docker run --name paddle_trt -i --rm ${CUDA_SO} ${DEVICES} -v $PWD:/paddle -w /paddle \
#    -e "CMAKE_BUILD_TYPE=RelWithDebInfo" \
#    -e "WITH_GPU=ON" \
#    -e "WITH_MKL=OFF" \
#    -e "WITH_AVX=ON" \
#    -e "CUDA_ARCH_NAME=Auto" \
#    -e "WITH_C_API=OFF" \
#    -e "WITH_TESTING=ON" \
#    -e "WITH_FLUID_ONLY=ON" \
#    -e "WITH_DEB=OFF" \
#    -e "WITH_ANAKIN=OFF" \
#    -e "RUN_TEST=OFF" \
#    -e "INSTALL_PREFIX=/paddle/dist" \
#    paddlepaddle/paddle:latest-dev \
#    paddle/scripts/paddle_build.sh build

#docker run --name paddle_trt --network=host -it --rm ${CUDA_SO} ${DEVICES} -v $PWD/../../Paddle:/paddle -v $PWD/../../inference:/data -w /paddle \
#    paddlepaddle/paddle:latest-dev \
#    build_paddle/build_docker.sh run

NAME=xreki/paddle
TAG=cuda92_cudnn7_debug
#NVIDIA_DOCKER_PATH=/home/liuyiqun/packages/nvidia-docker-2.0.3
nvidia-docker run --name paddle_xreki --network=host -it --rm \
    -v $PWD/../../Paddle:/paddle -v $PWD/../../inference:/data -v $PWD/../../models:/models -w /paddle \
    $NAME:$TAG \
    build_paddle/build_docker.sh $TASK

