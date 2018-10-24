#!/bin/bash
set -xe

NAME=$1
REPEAT=1
BATCH_SIZE=1

if [ $# -ge 2 ]; then
  REPEAT=$2
fi
if [ $# -ge 3 ]; then
  BATCH_SIZE=$3
fi

#DIRNAME=/home/liuyiqun01/PaddlePaddle/Paddle/python/paddle/fluid/tests/book/${NAME}.inference.model
#DIRNAME=/home/liuyiqun01/PaddlePaddle/Mobile/Demo/linux/image_classification/models/fluid/resnet50
#DIRNAME=/home/liuyiqun01/PaddlePaddle/Mobile/Demo/linux/image_classification/models/fluid/resnet50_merge_bn
#DIRNAME=/home/liuyiqun01/PaddlePaddle/inference/paddle_test/fluid_model/model
#DIRNAME=/home/liuyiqun01/PaddlePaddle/inference/mobilenet-ssd

export FLAGS_fraction_of_gpu_memory_to_use=0.1
#export GLOG_v=3
unset CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=3
#export FLAGS_benchmark=1
#./paddle/fluid/inference/tests/book/test_inference_${NAME} \
#    --dirname=${DIRNAME} \
#    --repeat=${REPEAT} \
#    --batch_size=${BATCH_SIZE}

DIRNAME=/data/InceptionV3_Model/InceptionV3_Model
IMAGE_PATH=/data/InceptionV3_Model/ocr_images/0_780.jpg.txt
/paddle/build_paddle/build_docker/paddle/fluid/inference/tests/api/ocr_plate \
    --dirname=${DIRNAME} \
    --image_path=${IMAGE_PATH} \
    --profile=1
