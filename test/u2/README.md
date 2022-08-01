
build

```
cmake -B build
cmake --build build
```

run

```
LD_LIBRARY_PATH=/workspace/DeepSpeech-2.x/tools/venv/lib/python3.7/site-packages/paddle/fluid:$LD_LIBRARY_PATH ./build/main
```
