cmake_minimum_required(VERSION 3.10)
project(A2)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)

set(HEADER_FILES functions.h)
set(SOURCE_FILES functions.cpp main.cpp)
add_executable(A2 ${SOURCE_FILES})

include_directories(${OpenCV_INCLUDE_DIRS})

target_link_libraries(A2 ${OpenCV_LIBS})