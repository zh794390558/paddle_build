mkdir -p build_cuda && cd build_cuda

# 执行cmake指令
cmake -DPY_VERSION=3.7 \
      -DWITH_TESTING=ON \
      -DWITH_DISTRIBUTE=ON \
      -DWITH_AVX=ON \
      -DWITH_MKL=ON \
      -DWITH_GPU=ON \
      -DCUDA_ARCH_NAME=Auto \
      -DWITH_TENSORRT=OFF \
      -DTENSORRT_ROOT=/usr/local/TensorRT-7.1.3.4/ \
      -DON_INFER=ON \
      -DCMAKE_BUILD_TYPE=Release \
      ..
# 使用make编译
make -j20
# 编译成功后可在dist目录找到生成的.whl包
#pip3 install python/dist/paddlepaddle-2.0.0-cp36-cp36m-linux_x86_64.whl
# 预测库编译
#make inference_lib_dist -j4
