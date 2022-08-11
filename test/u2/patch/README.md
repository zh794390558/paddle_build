* glog with gflags
when gfalgs installed, will case error like:

```
CMake Error at /workspace/Paddle/build_cuda/third_party/install/gflags/lib/cmake/gflags/gflags-nonamespace-targets.cmake:37 (message):
  Some (but not all) targets in this export set were already defined.

  Targets Defined: gflags_nothreads_static

  Targets not yet defined: gflags_static

Call Stack (most recent call first):
  /workspace/Paddle/build_cuda/third_party/install/gflags/lib/cmake/gflags/gflags-config.cmake:17 (include)
  fc_base/glog-src/CMakeLists.txt:51 (find_package)
```


* openfst link with gflags.
openfst as libs/flags.cc which impl gflags.
when openfst link with glfags, will case multidefintion error:

```
/workspace/u2/fc_base/gflags-build/libgflags_nothreads.a(gflags_reporting.cc.o):(.bss+0x0): multiple definition of `fLB::FLAGS_help'
.libs/flags.o:(.bss+0x4): first defined here/workspace/u2/fc_base/gflags-build/libgflags_nothreads.a(gflags_reporting.cc.o):(.bss+0x2): multiple definition of `fLB::FLAGS_helpshort'.libs/flags.o:(.bss+0x5): first defined here
/workspace/u2/fc_base/glog-build/libglog.a(vlog_is_on.cc.o):(.bss+0x0): multiple definition of `fLI::FLAGS_v'
.libs/flags.o:(.bss+0x0): first defined here
```