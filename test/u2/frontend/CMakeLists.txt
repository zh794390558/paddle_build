add_library(frontend STATIC
feature_pipeline.cc
fft.cc
cmvn.cc
)

# target_link_libraries(frontend PUBLIC utils glog)

# support graph feature pipeline, so add more deps.
target_compile_options(frontend PUBLIC ${PADDLE_COMPILE_FLAGS})
target_include_directories(frontend PUBLIC ${pybind11_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR})
target_link_libraries(frontend utils glog ${PYTHON_LIBRARIES} ${PADDLE_LINK_FLAGS})