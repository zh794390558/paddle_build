cmake_minimum_required(VERSION 3.14)
project(u2 LANGUAGES CXX)

find_package(Threads REQUIRED)
find_package (PythonLibs REQUIRED)
find_package (Python3 REQUIRED)
find_package(pybind11 CONFIG)

message(STATUS "PYTHON_LIBRARIES = ${PYTHON_LIBRARIES}")
message(STATUS "Python3_EXECUTABLE = ${Python3_EXECUTABLE}")
message(STATUS "Pybind11_INCLUDES = ${pybind11_INCLUDE_DIRS}, pybind11_LIBRARIES=${pybind11_LIBRARIES}, pybind11_DEFINITIONS=${pybind11_DEFINITIONS}")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_VERBOSE_MAKEFILE ON)

execute_process(COMMAND python -c "import paddle ; print(' '.join(paddle.sysconfig.get_link_flags()), end='')"
                OUTPUT_VARIABLE PADDLE_LINK_FLAGS
                RESULT_VARIABLE SUCESS)

message(STATUS PADDLE_LINK_FLAGS= ${PADDLE_LINK_FLAGS})
string(STRIP ${PADDLE_LINK_FLAGS} PADDLE_LINK_FLAGS)


execute_process(COMMAND python -c "import paddle ; print(' '.join(paddle.sysconfig.get_compile_flags()), end='')"
                OUTPUT_VARIABLE PADDLE_COMPILE_FLAGS)
message(STATUS PADDLE_COMPILE_FLAGS= ${PADDLE_COMPILE_FLAGS})
string(STRIP ${PADDLE_COMPILE_FLAGS} PADDLE_COMPILE_FLAGS)


set(name main_test)
add_executable(${name} main_test.cc)
target_compile_options(${name} PUBLIC ${PADDLE_COMPILE_FLAGS})
target_include_directories(${name} PUBLIC ${pybind11_INCLUDE_DIRS})
target_link_libraries(${name} ${PYTHON_LIBRARIES} ${PADDLE_LINK_FLAGS})

add_subdirectory(core)

add_executable(main main.cc)
target_link_libraries(main decoder)


enable_testing()
add_test(
    NAME main_test
    COMMAND $<TARGET_FILE:main_test>
)