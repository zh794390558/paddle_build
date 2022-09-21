#!/bin/bash

set -e


wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav

# u2pp 
if [ ! -d asr1_chunk_conformer_u2pp_wenetspeech_static_1.1.0.model ];then
   wget -c http://paddlespeech.bj.bcebos.com/s2t/wenetspeech/asr1/static/asr1_chunk_conformer_u2pp_wenetspeech_static_1.1.0.model.tar.gz
   tar zxvf asr1_chunk_conformer_u2pp_wenetspeech_static_1.1.0.model.tar.gz
fi


# u2
if [ ! -d asr1_chunk_conformer_u2_wenetspeech_static_1.1.0.model ];then
   wget -c http://paddlespeech.bj.bcebos.com/s2t/wenetspeech/asr1/static/asr1_chunk_conformer_u2_wenetspeech_static_1.1.0.model.tar.gz
   tar zxvf asr1_chunk_conformer_u2_wenetspeech_static_1.1.0.model.tar.gz
fi



