add_library(frontend STATIC
feature_pipeline.cc
fft.cc
cmvn.cc
)
target_link_libraries(frontend PUBLIC utils glog)