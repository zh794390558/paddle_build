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

. /paddle/build_paddle/image.sh
BUILD_ROOT=/paddle/build_paddle/build_docker${DOCKER_SUFFIX}

#DIRNAME=/home/liuyiqun01/PaddlePaddle/Paddle/python/paddle/fluid/tests/book/${NAME}.inference.model
#DIRNAME=/home/liuyiqun01/PaddlePaddle/Mobile/Demo/linux/image_classification/models/fluid/resnet50
#DIRNAME=/home/liuyiqun01/PaddlePaddle/Mobile/Demo/linux/image_classification/models/fluid/resnet50_merge_bn
#DIRNAME=/home/liuyiqun01/PaddlePaddle/inference/paddle_test/fluid_model/model
#DIRNAME=/home/liuyiqun01/PaddlePaddle/inference/mobilenet-ssd

export FLAGS_fraction_of_gpu_memory_to_use=0.1
#export GLOG_v=3
unset CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/paddle/TensorRT-4.0.1.6/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/paddle/gperftools-2.7/install/lib:$LD_LIBRARY_PATH
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
#MODEL_DIR=/data/face/models/super_res
#IMAGE_DIMS=1x3x207x175
#INPUT_NAME=image

#EXE_NAME=video_tester
#EXE_NAME=faster_rcnn_tester
#EXE_NAME=image_tester

MODEL_DIR=/data/ocr/1d-attention/models/origin
#MODEL_DIR=/data/ocr/1d-attention/models/opt_2
#MODEL_DIR=/data/ocr/1d-attention/models/opt_5
IMAGE_DIR=/data/ocr/1d-attention/case_txt
#IMAGE_DIR=/data/ocr/1d-attention/test_image_txt

EXE_NAME=ocr_plate_tester

if [ $XREKI_IMAGE_NAME == manylinux_trt ]; then
  patchelf --set-rpath /opt/compiler/gcc-4.8.2/lib64/ $BUILD_ROOT/paddle/fluid/inference/tests/api/samples/$EXE_NAME
  patchelf --set-interpreter /opt/compiler/gcc-4.8.2/lib64/ld-linux-x86-64.so.2 $BUILD_ROOT/paddle/fluid/inference/tests/api/samples/$EXE_NAME
fi

PPROF=0
if [ $PPROF -eq 1 ]; then
  /paddle/gperftools-2.7/install/bin/pprof --pdf \
      $BUILD_ROOT/paddle/fluid/inference/tests/api/samples/$EXE_NAME \
      paddle_inference.prof > $EXE_NAME.pdf
else
  $BUILD_ROOT/paddle/fluid/inference/tests/api/samples/$EXE_NAME \
      --infer_model=${MODEL_DIR} \
      --profile=0 \
      --repeat=1 \
      --prog_filename="model" \
      --param_filename="params" \
      --image_dims="1x0x0x0" \
      --image_dir=${IMAGE_DIR} \
      --use_gpu=1 \
      --use_analysis=0 \
      --use_tensorrt=0
fi


#cd $BUILD_ROOT
#make test ARGS="-R test_analyzer_resnet50 -V"

#$BUILD_ROOT/paddle/fluid/inference/tests/api/test_analyzer_resnet50 \
#    --infer_model=/paddle/build_paddle/resnet50_model
