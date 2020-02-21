cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(opencv_dnn_cuda_test_cpp)

set(EXECUTABLE_OUTPUT_PATH "..")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
find_package(OpenCV 4.2.0 REQUIRED)

add_executable(opencv_dnn_cuda_test_cpp main.cpp)

target_link_libraries(opencv_dnn_cuda_test_cpp ${OPENCV_LIBS})
