# Script files helping to compile PaddlePaddle for different platform

```bash
unset GREP_OPTIONS
export TMPDIR=/workspace/tmp
make
source venv/bin/activate
```

- Clone PaddlePaddle to somewhere
```bash
$ cd somewhere
somewhere $ git clone https://github.com/PaddlePaddle/Paddle.git
```

- Clone this repo under Paddle
```bash
somewhere $ cd Paddle
somewhere/Paddle $ git clone https://github.com/Xreki/build_paddle.git
somewhere/Paddle $ cd build_paddle
```

- Build PaddlePaddle for linux
```bash
somewhere/Paddle/build_paddle $ ./build.sh
```

- Build PaddlePaddle for Mac
```bash
somewhere/Paddle/build_paddle $ ./build_mac.sh
```

- Build PaddlePaddle for Android
```bash
somewhere/Paddle/build_paddle $ ./build_android.sh 64 # arm64-v8a
somewhere/Paddle/build_paddle $ ./build_android.sh 32 # armeabi-v7a
somewhere/Paddle/build_paddle $ ./build_android.sh # armeabi
```

- Build PaddlePaddle for iOS
```bash
somewhere/Paddle/build_paddle $ ./build_ios.sh arm64
somewhere/Paddle/build-paddle $ ./build_ios.sh armv7
```
## Reference
* https://github.com/Xreki/build_paddle.git
