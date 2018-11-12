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

#SUFFIX=_cpu
BUILD_ROOT=/paddle/build_paddle/build_docker$SUFFIX

#DIRNAME=/home/liuyiqun01/PaddlePaddle/Paddle/python/paddle/fluid/tests/book/${NAME}.inference.model
#DIRNAME=/home/liuyiqun01/PaddlePaddle/Mobile/Demo/linux/image_classification/models/fluid/resnet50
#DIRNAME=/home/liuyiqun01/PaddlePaddle/Mobile/Demo/linux/image_classification/models/fluid/resnet50_merge_bn
#DIRNAME=/home/liuyiqun01/PaddlePaddle/inference/paddle_test/fluid_model/model
#DIRNAME=/home/liuyiqun01/PaddlePaddle/inference/mobilenet-ssd

export FLAGS_fraction_of_gpu_memory_to_use=0.1
#export GLOG_v=3
unset CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=7
#export FLAGS_benchmark=1
#./paddle/fluid/inference/tests/book/test_inference_${NAME} \
#    --dirname=${DIRNAME} \
#    --repeat=${REPEAT} \
#    --batch_size=${BATCH_SIZE}

#MODEL_DIR=/data/InceptionV3_Model/InceptionV3_Model
#DATA_PATH=/data/InceptionV3_Model/ocr_images/0_780.jpg.txt

#MODEL_DIR=/data/video/VLAD_3s/stage1
#MODEL_DIR=/data/video/fluid_lstm_first_v6_transmodel
#MODEL_DIR=/data/video/Attention_3s/stage1
#DATA_DIR=/data/video/data
#MODEL_DIR=/data/faster_rcnn/faster_rcnn_resnet50
#MODEL_DIR=/data/face/models/blur
#IMAGE_DIMS=1x3x112x112
#MODEL_DIR=/data/face/models/emotion
#IMAGE_DIMS=1x3x144x128
#INPUT_NAME=data
#MODEL_DIR=/paddle/build_paddle/resnet50_model #image_dims=1x3x318x318

#MODEL_DIR=/data/face/models/demark
MODEL_DIR=/data/face/models/super_res
IMAGE_DIMS=1x3x207x175
INPUT_NAME=image

#EXE_NAME=video_tester
#EXE_NAME=ocr_plate_tester
#EXE_NAME=faster_rcnn_tester
EXE_NAME=image_tester

$BUILD_ROOT/paddle/fluid/inference/tests/api/$EXE_NAME \
    --model_dir=${MODEL_DIR} \
    --profile=1 \
    --repeat=1000 \
    --use_gpu=1 \
    --image_dims=${IMAGE_DIMS} \
    --input_name=${INPUT_NAME} \
    --use_tensorrt=1

#--data_path=${DATA_PATH} \

#cd $BUILD_ROOT
#make test ARGS="-R test_analyzer_resnet50 -V"

#$BUILD_ROOT/paddle/fluid/inference/tests/api/test_analyzer_resnet50 \
#    --infer_model=/paddle/build_paddle/resnet50_model
