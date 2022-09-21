# Streaming Conformer U2

## build

```
cmake -B build
cmake --build build
```

run

```
LD_LIBRARY_PATH=/workspace/DeepSpeech-2.x/tools/venv/lib/python3.7/site-packages/paddle/fluid:/workspace/DeepSpeech-2.x/tools/venv/lib/python3.7/site-packages/paddle/libs/:$LD_LIBRARY_PATH
```


## Run

```
./local/download.sh
./local/run_wav.sh
```


## w/o kaldi
2d0f5d844badd3a6ff569cd7deef0f87da2d794e


## test wav

wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav


## vocab to unit

```
awk '{print $0 , " ",  NR-1}' vocab.txt 
```

## BiDecoder or UiDecoder
reverse_weight is 0.0 is UiDecoder, other wise is BiDecoder


## Tracing View

chrome://tracing/
