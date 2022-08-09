
set(decoder_srcs
asr_itf.cc
pd_asr_model.cc
ctc_prefix_beam_search.cc
)

add_library(decoder STATIC ${decoder_srcs})
target_compile_options(decoder PUBLIC ${PADDLE_COMPILE_FLAGS})
target_include_directories(decoder PUBLIC ${pybind11_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR})
target_link_libraries(decoder ${PYTHON_LIBRARIES} ${PADDLE_LINK_FLAGS})