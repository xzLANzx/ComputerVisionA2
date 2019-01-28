cmake_minimum_required(VERSION 3.10)
project(A2)

set(CMAKE_CXX_STANDARD 11)

add_executable(A2 main.cpp)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(A2 ${OpenCV_LIBS})