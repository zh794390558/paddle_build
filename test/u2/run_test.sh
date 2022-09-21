#!/bin/bash

set -e


if [ ! -d asr1_chunk_conformer_u2pp_wenetspeech_static_1.1.0.model ];then
   wget -c http://paddlespeech.bj.bcebos.com/s2t/wenetspeech/asr1/asr1_chunk_conformer_u2pp_wenetspeech_static_1.1.0.model.tar.gz
   tar zxvf asr1_chunk_conformer_u2pp_wenetspeech_static_1.1.0.model.tar.gz
fi

export LD_LIBRARY_PATH=/workspace/DeepSpeech-2.x/tools/venv/lib/python3.7/site-packages/paddle/fluid:/workspace/DeepSpeech-2.x/tools/venv/lib/python3.7/site-packages/paddle/libs/:$LD_LIBRARY_PATH

./build/main_test

#./build/main

# FLAG_logbuflevel=-1 GLOG_logtostderr=1 GLOG_v=3 ./run.sh 
#./build/decoder_main \
#	--model_path "asr1_chunk_conformer_u2pp_wenetspeech_static_1.1.0.model/export.jit" \
#	--unit_path "asr1_chunk_conformer_u2pp_wenetspeech_static_1.1.0.model/unit.txt" \
#	--cmvn_path "asr1_chunk_conformer_u2pp_wenetspeech_static_1.1.0.model/mean_std.json" \
#	--wav_path zh.wav
