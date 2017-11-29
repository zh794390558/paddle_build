#!/bin/sh

ARG="$1"
NAME=paddle-android
FILE=Dockerfile.android

if [ $# -lt 1 ]; then
  echo "Specify: build or run"
  exit
fi

if [ $# -ge 2 ]; then
  if [ $2 -eq 32 ]; then
    ANDROID_ABI="armeabi-v7a"
  elif [ $2 -eq 64 ]; then
    ANDROID_ABI="arm64-v8a"
  else
    echo "Build for armeabi default"
    ANDROID_ABI="armeabi"
  fi
else
  ANDROID_ABI="armeabi"
fi

if [ $# -eq 3 ]; then
  ANDROID_API=$3
else
  ANDROID_API=21
fi

if [ $ARG == "build" ]; then
  docker build -t xreki/$NAME:dev . -f $FILE
elif [ $ARG == "run" ]; then
  docker run -it --rm -v $PWD:/paddle -e "ANDROID_ABI=$ANDROID_ABI" -e "ANDROID_API=$ANDROID_API" xreki/$NAME:dev #bash /paddle/build_paddle/build_android_docker.sh
fi
