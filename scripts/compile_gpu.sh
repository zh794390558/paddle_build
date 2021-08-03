#!/bin/bash

pip3.7 install protobuf
apt install patchelf

cmake -B build -DPY_VERSION=3.7 -DWITH_GPU=ON -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)
