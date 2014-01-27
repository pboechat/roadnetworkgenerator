cmake_minimum_required(VERSION 2.8)

project(roadnetworkgenerator)

option(USE_QUADTREE "Use quadtree to minimize the number of collision tests on edges added to graph" TRUE)
option(USE_CUDA "" FALSE)

add_subdirectory(dependencies)
add_subdirectory(src)
add_subdirectory(test)