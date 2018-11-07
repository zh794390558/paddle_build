#!/bin/sh

export SIZE_X=128
export SIZE_Y=128
export SIZE_Z=128
export AXIS=2

#export GLOG_v=3
unset CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=2
#FLAGS_fraction_of_gpu_memory_to_use=0.16 make test ARGS="-R $1 -V"
make test ARGS="-R $1 -V"
