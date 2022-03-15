cmake_minimum_required(VERSION 2.8)
project(facere)
set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)

message(STATUS ${OpenCV_LIBRARY_DIRS})

include_directories(
        /usr/include/freetype2/
        ${OpenCV_INCLUDE_DIRS}
        )

#link_directories(${OpenCV_LIBRARY_DIRS} )

add_executable(facere facere.cpp CvxText.cpp CvxText.h)

target_link_libraries(facere ${OpenCV_LIBS} freetype)


