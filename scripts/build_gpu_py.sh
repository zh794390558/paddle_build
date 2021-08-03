mkdir build_cuda && cd build_cuda

# 执行cmake指令
cmake -DPY_VERSION=3 \
      -DWITH_TESTING=OFF \
      -DWITH_MKL=ON \
      -DWITH_GPU=ON \
      -DWITH_TENSORRT=ON \
      -DTENSORRT_ROOT=/home/work/nvidia/TensorRT-7.1.3.4 \
      -DON_INFER=ON \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      ..
# 使用make编译
make -j4
# 编译成功后可在dist目录找到生成的.whl包
pip3 install python/dist/paddlepaddle-2.0.0-cp36-cp36m-linux_x86_64.whl
# 预测库编译
make inference_lib_dist -j4
