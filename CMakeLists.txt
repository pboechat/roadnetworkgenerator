cmake_minimum_required(VERSION 2.8)

project(roadnetworkgenerator)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

find_package(OpenGL REQUIRED)
find_package(CUDA_DLINK REQUIRED)

# =================
# 	CUDA OPTIONS
# =================

option(CUDA_BUILD_CC20 "Build with compute capability 2.0 support" FALSE)
option(CUDA_BUILD_CC30 "Build with compute capability 3.0 support" TRUE)
option(CUDA_BUILD_CC35 "Build with compute capability 3.5 support" FALSE)
option(CUDA_BUILD_INFO "Build with kernel statistics and line numbers" FALSE)
option(CUDA_BUILD_DEBUG "Build with kernel debug" TRUE)

set(CUDA_NVCC_FLAGS "-use_fast_math;")

if(CUDA_BUILD_CC20)
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-gencode=arch=compute_20,code=sm_20")
endif()

if(CUDA_BUILD_CC30)
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-gencode=arch=compute_30,code=sm_30")
endif()

if(CUDA_BUILD_CC35)
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-gencode=arch=compute_35,code=sm_35;-maxrregcount=64")
endif()

if(CUDA_BUILD_INFO)
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-keep;--ptxas-options=-v;-lineinfo")
endif()

if(CUDA_BUILD_DEBUG)
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-G")
endif()

# ===========================
# PROJECT
# ===========================

file(GLOB CPP_HEADERS include/*.h)
file(GLOB CPP_SOURCES src/*.cpp src/*.c)
file(GLOB CUDA_HEADERS include/*.cuh)
file(GLOB CUDA_SOURCES src/*.cu)
file(GLOB SHADERS shaders/*.*)

include_directories(include/
					${OPENGL_INCLUDE}
					${CUDA_INCLUDE_DIRS}
					${CMAKE_SOURCE_DIR}/dependencies/include
)

source_group("CUDA Header Files" FILES ${CUDA_HEADERS})
source_group("CUDA Source Files" FILES ${CUDA_SOURCES})
source_group("Shader Files" FILES ${SHADERS})

cuda_add_executable(
	roadnetworkgenerator
	${CPP_HEADERS}
	${CPP_SOURCES}
	${CUDA_HEADERS}
	${CUDA_SOURCES}
	${SHADERS}
)

add_custom_command(TARGET roadnetworkgenerator POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory 
	"${PROJECT_SOURCE_DIR}/dependencies/libs" $<TARGET_FILE_DIR:roadnetworkgenerator>
)

target_link_libraries(
	roadnetworkgenerator
	${OPENGL_LIBRARY}
	${OPENGL_gl_LIBRARY}
	${OPENGL_glu_LIBRARY}
	${CMAKE_SOURCE_DIR}/dependencies/libs/FreeImage.lib
)
