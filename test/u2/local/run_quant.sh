#!/bin/bash

set -x
set -e

export LD_LIBRARY_PATH=/workspace/DeepSpeech-2.x/tools/venv/lib/python3.7/site-packages/paddle/fluid:/workspace/DeepSpeech-2.x/tools/venv/lib/python3.7/site-packages/paddle/libs/:$LD_LIBRARY_PATH


# FLAG_logbuflevel=-1 GLOG_logtostderr=1 GLOG_v=3 ./run.sh 
#model_dir=asr1_chunk_conformer_u2_wenetspeech_static_1.1.0.model
#reverse_weight=0.0

#model_dir=asr1_chunk_conformer_u2pp_wenetspeech_static_1.1.0.model
model_dir=asr1_chunk_conformer_u2pp_wenetspeech_static_quant_1.1.0.model/
model_prefix=$model_dir/export
reverse_weight=0.3

chunk_size=16
./build/decoder_main \
        --feature_pipeline_type kaldi \
        --reverse_weight $reverse_weight \
        --chunk_size $chunk_size \
        --rescoring_weight 1.0 \
	--model_path $model_prefix \
	--unit_path "$model_dir/unit.txt" \
	--cmvn_path "$model_dir/mean_std.json" \
        --result exp/wav.aishell.test.chunk16.hyp \
	--wav_scp data/wav.20.scp
	#--wav_scp data/wav.aishell.test.scp
	#--wav_path zh.wav 
