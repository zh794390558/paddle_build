


#gigaspeech
# wget -c https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/gigaspeech/20210728_u2pp_conformer_libtorch.tar.gz

# wenetspeech
wget -c https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/wenetspeech/20220506_u2pp_conformer_libtorch.tar.gz

if [ ! -d 20220506_u2pp_conformer_libtorch/ ]; then
	tar zxvf 20220506_u2pp_conformer_libtorch.tar.gz
fi


# wget -c https://paddlespeech.bj.bcebos.com/s2t/paddle_asr_online/aishell_test.zip

wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav
wav_path=zh.wav
#model_dir=20210728_u2pp_conformer_libtorch/
#model_dir=20210601_u2++_conformer_libtorch/
model_dir=20220506_u2pp_conformer_libtorch/

export GLOG_logtostderr=1
export GLOG_v=2
#./build/decoder_main \
#	--chunk_size 16 \
#	--wav_path $wav_path \
#	--model_path $model_dir/final.zip \
#	--dict_path $model_dir/units.txt 2>&1 | tee log.txt

wav_scp=wav.aishell.test.scp
#wav_scp=wav.scp
result=wav.aishell.test.hyp
dict_path=$model_dir/units.txt
./build/decoder_main \
	--chunk_size 16 \
	--wav_scp $wav_scp \
	--result $result \
	--model_path $model_dir/final.zip \
	--dict_path $dict_path 2>&1 | tee log.txt
