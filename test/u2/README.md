# Streaming Conformer U2



## env
* ubuntu 16.04
* python3.7
* build paddlepaddle from sourse with `-DWITH_AVX=ON -DWITH_MKL=ON -DWITH_MKLDNN=ON`
* cp `sysconfig.bak` into `/workspace/DeepSpeech-2.x/tools/venv/lib/python3.7/site-packages/paddle/sysconfig.py`

```
Architecture:        x86_64
CPU op-mode(s):      32-bit, 64-bit
Byte Order:          Little Endian
CPU(s):              64
On-line CPU(s) list: 0-63
Thread(s) per core:  2
Core(s) per socket:  32
Socket(s):           1
NUMA node(s):        1
Vendor ID:           GenuineIntel
CPU family:          6
Model:               85
Model name:          Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz
Stepping:            4
CPU MHz:             2394.374
BogoMIPS:            4788.74
Hypervisor vendor:   KVM
Virtualization type: full
L1d cache:           32K
L1i cache:           32K
L2 cache:            1024K
L3 cache:            28160K
NUMA node0 CPU(s):   0-63
Flags:               fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss ht syscall nx pdpe1gb rdtscp lm constant_tsc arch_perfmon rep_good nopl xtopology eagerfpu pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand hypervisor lahf_lm abm 3dnowprefetch invpcid_single ibrs ibpb fsgsbase tsc_adjust bmi1 hle avx2 smep bmi2 erms invpcid rtm mpx avx512f avx512dq rdseed adx smap clflushopt clwb avx512cd avx512bw avx512vl xsaveopt xsavec xgetbv1 arat umip pku ospke spec_ctrl
```
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

## Test Data

Test data format is like `data/wav.aishell.test.scp`, data is download from `https://paddlespeech.bj.bcebos.com/s2t/paddle_asr_online/aishell_test.zip`.

You can prepare `wav.scp` like below:
```
wget -c https://paddlespeech.bj.bcebos.com/s2t/paddle_asr_online/aishell_test.zip
unzip  aishell_test.zip

realpath $data/test/*/*.wav > $data/wavlist
awk -F '/' '{ print $(NF) }' $data/wavlist | awk -F '.' '{ print $1 }' > $data/utt_id
paste $data/utt_id $data/wavlist > $data/$aishell_wav_scp
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
