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

#export FLAGS_fraction_of_gpu_memory_to_use=0.1
#export GLOG_v=30
unset CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=7
#export FLAGS_benchmark=1
#./paddle/fluid/inference/tests/book/test_inference_${NAME} \
#    --dirname=${DIRNAME} \
#    --repeat=${REPEAT} \
#    --batch_size=${BATCH_SIZE}


# video models
#MODEL_DIR=/data/video/VLAD_3s/stage1
#MODEL_DIR=/data/video/fluid_lstm_first_v6_transmodel
#MODEL_DIR=/data/video/Attention_3s/stage1
#DATA_DIR=/data/video/data

#MODEL_DIR=/data/faster_rcnn/faster_rcnn_resnet50
#EXE_NAME=faster_rcnn_tester
#$BUILD_ROOT/paddle/fluid/inference/tests/api/$EXE_NAME \
#    --model_dir=${MODEL_DIR} \
#    --profile=1 \
#    --repeat=100 \
#    --use_gpu=1 \
#    --use_tensorrt=0

# ocr models
#MODEL_DIR=/data/ocr/InceptionV3_Model/InceptionV3_Model
#DATA_PATH=/data/InceptionV3_Model/ocr_images/0_780.jpg.txt
#MODEL_DIR=/data/ocr/1d-attention/models/origin
MODEL_DIR=/data/ocr/1d-attention/models/opt_1

#MODEL_DIR=/data/face/models/blur
#IMAGE_PATH=/data/face/images/blur.txt
#MODEL_DIR=/data/face/models/emotion
#MODEL_DIR=/data/face/models/demark
#MODEL_DIR=/data/face/models/super_res
#MODEL_DIR=/data/face/models/pars

#EXE_NAME=video_tester
EXE_NAME=ocr_plate_tester
$BUILD_ROOT/paddle/fluid/inference/tests/api/samples/$EXE_NAME \
    --infer_model=${MODEL_DIR} \
    --profile=0 \
    --repeat=100 \
    --prog_filename="model" \
    --param_filename="params" \
    --use_analysis=0 \
    --use_tensorrt=0


#cd $BUILD_ROOT
#make test ARGS="-R test_trt_elementwise_op -V"

#$BUILD_ROOT/paddle/fluid/inference/tests/api/test_analyzer_resnet50 \
#    --infer_model=/paddle/build_paddle/resnet50_model
#MODEL_DIR=/data/facebox_model_remove_ops
#MODEL_DIR=/paddle/build_paddle/resnet50_model #image_dims=1x3x318x318
#MODEL_DIR=/paddle/build_paddle/trt_test_models
#MODEL_DIR=/data/se_resnext_50/se_resnext
#MODEL_DIR=/data/se_resnext_50/models
#$BUILD_ROOT/paddle/fluid/inference/tests/api/test_trt_models \
#    --infer_model=${MODEL_DIR} \
#    --profile=0 \
#    --use_tensorrt=1 \
#    --prog_filename="model" \
#    --param_filename="params" \
#    --repeat=100

