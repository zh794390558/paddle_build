#!/bin/sh

set -xe

export CUDA_SO="$(\ls /usr/lib64/libcuda* | xargs -I{} echo '-v {}:{}') $(\ls /usr/lib64/libnvidia* | xargs -I{} echo '-v {}:{}')"
export DEVICES=$(\ls /dev/nvidia* | xargs -I{} echo '--device {}:{}')

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

docker run --name paddle_trt -i --rm ${CUDA_SO} ${DEVICES} -v $PWD/../../Paddle:/paddle -v $PWD/../../inference:/data -w /paddle \
    paddlepaddle/paddle:latest-dev \
    build_paddle/build_docker.sh build
