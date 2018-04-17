#!/bin/sh

#export GLOG_v=3
unset CUDA_VISIBLE_DEVICES
#export CUDA_VISIBLE_DEVICES=0
#FLAGS_fraction_of_gpu_memory_to_use=0.16 make test ARGS="-R $1 -V"
make test ARGS="-R $1 -V"
