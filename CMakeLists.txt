cmake_minimum_required(VERSION 2.8)

project(roadnetworkgenerator)

option(USE_QUADTREE "Use quadtree to minimize the number of collision tests on edges added to graph" TRUE)

add_subdirectory(src)