set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

find_package(CUDA_DLINK REQUIRED)

option(CUDA_BUILD_CC20 "Build with compute capability 2.0 support" FALSE)
option(CUDA_BUILD_CC30 "Build with compute capability 3.0 support" TRUE)
option(CUDA_BUILD_CC35 "Build with compute capability 3.5 support" FALSE)
option(CUDA_BUILD_INFO "Build with kernel statistics and line numbers" FALSE)
option(CUDA_BUILD_DEBUG "Build with kernel debug" TRUE)
option(CUDA_BUILD_MAXRREGCOUNT_32 "Build with maxrregcount set to 32" FALSE)

set(CUDA_NVCC_FLAGS "-use_fast_math;-w")

if(CUDA_BUILD_CC20)
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-gencode=arch=compute_20,code=sm_20")
	add_definitions(/D "CUDA_CC=20")
	add_definitions(-DCUDA_CC=20)
endif()

if(CUDA_BUILD_CC30)
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-gencode=arch=compute_30,code=sm_30")
	add_definitions(/D "CUDA_CC=30")
	add_definitions(-DCUDA_CC=30)
endif()

if(CUDA_BUILD_CC35)
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-gencode=arch=compute_35,code=sm_35")
	add_definitions(/D "CUDA_CC=35")
	add_definitions(-DCUDA_CC=35)
endif()

if(CUDA_BUILD_INFO)
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-keep;--ptxas-options=-v;-lineinfo")
endif()

if(CUDA_BUILD_DEBUG)
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-G")
endif()

if(CUDA_BUILD_MAXRREGCOUNT_32)
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-maxrregcount=32")
endif()
	
find_package(OpenGL REQUIRED)

file(GLOB CPP_HEADERS *.h)
file(GLOB CPP_SOURCES *.cpp)
file(GLOB CUDA_HEADERS *.cuh)
file(GLOB CUDA_SOURCES *.cu)
file(GLOB SHADERS ${CMAKE_SOURCE_DIR}/shaders/*.glsl)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}
					${OPENGL_INCLUDE}
					${CMAKE_SOURCE_DIR}/dependencies/glm/include
					${CMAKE_SOURCE_DIR}/dependencies/cutil/include
					${CMAKE_SOURCE_DIR}/dependencies/FreeImage/include
					${CMAKE_SOURCE_DIR}/dependencies/gl3w/include
					${CMAKE_SOURCE_DIR}/dependencies/GL_utils/include
					${CMAKE_SOURCE_DIR}/dependencies/glFont/include
					${CMAKE_SOURCE_DIR}/dependencies/vld/include
)

source_group("Shader Files" FILES ${SHADERS})
source_group("CUDA Headers" FILES ${CUDA_HEADERS})
source_group("CUDA Sources" FILES ${CUDA_SOURCES})

if (MSVC)
	add_definitions(/D "_CRT_SECURE_NO_WARNINGS")
endif()

add_definitions(/D "USE_GLM")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-DUSE_GLM")

if (PARALLEL)
	add_definitions(/D "PARALLEL")
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-DPARALLEL")
endif()

if (COLLECT_STATISTICS)
	add_definitions(/D "COLLECT_STATISTICS")
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-DCOLLECT_STATISTICS")
endif()

if (PERSISTENT_THREADS)
	add_definitions(/D "PERSISTENT_THREADS")
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-DPERSISTENT_THREADS")
endif()

cuda_add_executable(
	roadnetworkgenerator
	${CPP_HEADERS}
	${CPP_SOURCES}
	gl3w.c
	${CUDA_HEADERS}
	${CUDA_SOURCES}
	${SHADERS}
)

add_custom_command(TARGET roadnetworkgenerator POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory 
	"${PROJECT_SOURCE_DIR}/dependencies/FreeImage/lib" $<TARGET_FILE_DIR:roadnetworkgenerator>
)

add_custom_command(TARGET roadnetworkgenerator POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory 
	"${PROJECT_SOURCE_DIR}/dependencies/FreeImage/bin" $<TARGET_FILE_DIR:roadnetworkgenerator>
)

target_link_libraries(
	roadnetworkgenerator
	${OPENGL_LIBRARY}
	${OPENGL_gl_LIBRARY}
	${OPENGL_glu_LIBRARY}
	${CMAKE_SOURCE_DIR}/dependencies/FreeImage/lib/FreeImage.lib
	${CMAKE_SOURCE_DIR}/dependencies/vld/lib/vld.lib
	${CUDA_curand_LIBRARY}
)

set_target_properties(roadnetworkgenerator PROPERTIES DEBUG_POSTFIX d)
