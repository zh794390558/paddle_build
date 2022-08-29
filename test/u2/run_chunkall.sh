#!/bin/bash

set -e


if [ ! -d chunk_wenetspeech_static ];then
   wget -c http://paddlespeech.bj.bcebos.com/s2t/wenetspeech/asr1/chunk_wenetspeech_static.tar.gz
   tar zxvf chunk_wenetspeech_static.tar.gz
fi

export LD_LIBRARY_PATH=/workspace/DeepSpeech-2.x/tools/venv/lib/python3.7/site-packages/paddle/fluid:/workspace/DeepSpeech-2.x/tools/venv/lib/python3.7/site-packages/paddle/libs/:$LD_LIBRARY_PATH

#./build/main_test

#./build/main

# FLAG_logbuflevel=-1 GLOG_logtostderr=1 GLOG_v=3 ./run.sh 
./build/decoder_main \
        --chunk_size -1 \
	--model_path "chunk_wenetspeech_static/export.jit" \
	--unit_path "chunk_wenetspeech_static/unit.txt" \
	--cmvn_path "chunk_wenetspeech_static/mean_std.json" \
	--wav_path zh.wav
