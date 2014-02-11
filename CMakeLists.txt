cmake_minimum_required(VERSION 2.8)

project(roadnetworkgenerator)

option(USE_CUDA "Run road network generation on GPU using CUDA" FALSE)
option(COLLECT_STATISTICS "Collect statistics about memory consumption and others" FALSE)

add_subdirectory(dependencies)
add_subdirectory(src)
add_subdirectory(test)