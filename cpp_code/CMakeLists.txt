cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(opencv_dnn_cuda_test_cpp)

set(EXECUTABLE_OUTPUT_PATH "..")

find_package(OpenCV REQUIRED)

add_executable(opencv_dnn_cuda_test_cpp main.cpp)

target_link_libraries(opencv_dnn_cuda_test_cpp "${OPENCV_LIBS}")

set_property(TARGET opencv_dnn_cuda_test_cpp PROPERTY CXX_STANDARD 11)
