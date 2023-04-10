#!/bin/bash

set -e

# FLAG_logbuflevel=-1 GLOG_logtostderr=1 GLOG_v=3 ./run.sh 
#model_dir=asr1_chunk_conformer_u2_wenetspeech_static_1.1.0.model
#reverse_weight=0.0

model_dir=asr1_chunk_conformer_u2pp_wenetspeech_static_1.3.0.model
reverse_weight=0.3

chunk_size=16

export DNNL_VERBOSE=0
export OMP_NUM_THREADS=1
./build/decoder_main \
        --feature_pipeline_type kaldi \
        --reverse_weight $reverse_weight \
        --chunk_size $chunk_size \
        --rescoring_weight 1.0 \
	--model_path "$model_dir/export.jit" \
	--unit_path "$model_dir/unit.txt" \
	--cmvn_path "$model_dir/mean_std.json" \
	--wav_path zh.wav 
