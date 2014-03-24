cmake_minimum_required(VERSION 2.8)

project(roadnetworkgenerator)

option(PARALLEL "Run roadnetwork generation in parallel on GPU using CUDA" FALSE)
option(COLLECT_STATISTICS "Collect statistics about memory consumption and others" FALSE)
option(PERSISTENT_THREADS "Use persistent threads technique on roadnetwork expansion kernel" FALSE)

add_subdirectory(dependencies)
add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(statistics)
add_subdirectory(presentation)
