cmake_minimum_required(VERSION 3.14)
project(u2 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_VERBOSE_MAKEFILE ON)


option(USE_TEST "whether to build unit test" ON)

# third party
include(FetchContent)
include(ExternalProject)
set(FETCHCONTENT_QUIET OFF)
get_filename_component(fc_base "fc_base" REALPATH BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(FETCHCONTENT_BASE_DIR ${fc_base})

# cmake
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)


# debug mode
if(CMAKE_BUILD_TYPES STREQUAL Debug)
    add_definitions(-DDEUBG)
endif()

# openfst
if(NOT MSVC)
  # Keep the same with openfst, -fPIC or -fpic
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread -fPIC")
else()
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
  add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
endif()

# reruied packages
find_package(Threads REQUIRED)
find_package (PythonLibs REQUIRED)
find_package (Python3 REQUIRED)
find_package(pybind11 CONFIG)

message(STATUS "PYTHON_LIBRARIES = ${PYTHON_LIBRARIES}")
message(STATUS "Python3_EXECUTABLE = ${Python3_EXECUTABLE}")
message(STATUS "Pybind11_INCLUDES = ${pybind11_INCLUDE_DIRS}, pybind11_LIBRARIES=${pybind11_LIBRARIES}, pybind11_DEFINITIONS=${pybind11_DEFINITIONS}")

# paddle include and compile option
execute_process(COMMAND python -c "import paddle ; print(' '.join(paddle.sysconfig.get_link_flags()), end='')"
                OUTPUT_VARIABLE PADDLE_LINK_FLAGS
                RESULT_VARIABLE SUCESS)

message(STATUS PADDLE_LINK_FLAGS= ${PADDLE_LINK_FLAGS})
string(STRIP ${PADDLE_LINK_FLAGS} PADDLE_LINK_FLAGS)

execute_process(COMMAND python -c "import paddle ; print(' '.join(paddle.sysconfig.get_compile_flags()), end='')"
                OUTPUT_VARIABLE PADDLE_COMPILE_FLAGS)
message(STATUS PADDLE_COMPILE_FLAGS= ${PADDLE_COMPILE_FLAGS})
string(STRIP ${PADDLE_COMPILE_FLAGS} PADDLE_COMPILE_FLAGS)


set(PADDLE_LIB_DIRS /workspace/DeepSpeech-2.x/tools/venv/lib/python3.7/site-packages/paddle/fluid:/workspace/DeepSpeech-2.x/tools/venv/lib/python3.7/site-packages/paddle/libs/)


# Include all dependency
include(openfst)
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
)


# submodule
add_subdirectory(utils)
add_dependencies(utils openfst)
add_subdirectory(frontend)
add_subdirectory(decoder)
add_dependencies(decoder openfst)


# bins
add_executable(main main.cc)
target_link_libraries(main decoder utils)

add_executable(decoder_main decoder_main.cc)
target_link_libraries(decoder_main decoder utils frontend fst)

# test bins
set(name main_test)
add_executable(${name} main_test.cc)
target_compile_options(${name} PUBLIC ${PADDLE_COMPILE_FLAGS})
target_include_directories(${name} PUBLIC ${pybind11_INCLUDE_DIRS})
target_link_libraries(${name} ${PYTHON_LIBRARIES} ${PADDLE_LINK_FLAGS})

if(USE_TEST)
    enable_testing()
    include(gtest)
    add_subdirectory(test)

    # add_test(
    #     NAME main_test
    #     COMMAND $<TARGET_FILE:main_test>
    # )
endif()