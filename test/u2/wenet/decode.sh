#./build/decoder_main     --chunk_size -1     --wav_path $wav_path     --model_path $model_dir/final.zip     --dict_path $model_dir/words.txt 2>&1 | tee log.txt


chunk_size=-1
ctc_weight=0.5
reverse_weight=0.0
rescoring_weight=1.0


# For CTC WFST based decoding
fst_path=./TLG.fst
acoustic_scale=1.0
beam=15.0
lattice_beam=7.5
min_active=200
max_active=7000
blank_skip_thresh=0.98

wfst_decode_opts="--fst_path $fst_path"
wfst_decode_opts="$wfst_decode_opts --beam $beam"
wfst_decode_opts="$wfst_decode_opts --lattice_beam $lattice_beam"
wfst_decode_opts="$wfst_decode_opts --max_active $max_active"
wfst_decode_opts="$wfst_decode_opts --min_active $min_active"
wfst_decode_opts="$wfst_decode_opts --acoustic_scale $acoustic_scale"
wfst_decode_opts="$wfst_decode_opts --blank_skip_thresh $blank_skip_thresh"
echo $wfst_decode_opts 


dir=exp
wav_scp=wav.scp
model_file=20210601_u2++_conformer_libtorch/final.zip
dict_file=20210601_u2++_conformer_libtorch/words.txt



#  # 7.4 Decoding with runtime
#  chunk_size=-1
#  ./tools/decode.sh --nj 16 \
#    --beam 15.0 --lattice_beam 7.5 --max_active 7000 \
#    --blank_skip_thresh 0.98 --ctc_weight 0.5 --rescoring_weight 1.0 \
#    --chunk_size $chunk_size \
#    --fst_path data/lang_test/TLG.fst \
#    data/test/wav.scp data/test/text $dir/final.zip \
#    data/lang_test/words.txt $dir/lm_with_runtime
#  # Please see $dir/lm_with_runtime for wer



./build/decoder_main \
     --rescoring_weight $rescoring_weight \
     --ctc_weight $ctc_weight \
     --reverse_weight $reverse_weight \
     --chunk_size $chunk_size \
     --wav_path ./zh.wav \
     --model_path $model_file \
     --dict_path $dict_file \
     $wfst_decode_opts \
     --result ${dir}/rsl.text &> ${dir}/rsl.log

#./build/decoder_main \
#     --rescoring_weight $rescoring_weight \
#     --ctc_weight $ctc_weight \
#     --reverse_weight $reverse_weight \
#     --chunk_size $chunk_size \
#     --wav_scp $wav_scp \
#     --model_path $model_file \
#     --dict_path $dict_file \
#     $wfst_decode_opts \
#     --result ${dir}/rsl.text &> ${dir}/rsl.log
