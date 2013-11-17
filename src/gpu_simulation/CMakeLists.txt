set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

find_package(OpenGL REQUIRED)
find_package(CUDA_DLINK REQUIRED)

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

file(GLOB CPP_HEADERS *.h)
file(GLOB CPP_SOURCES *.cpp *.c)
file(GLOB CUDA_HEADERS *.cuh)
file(GLOB CUDA_SOURCES *.cu)
file(GLOB SHADERS ${CMAKE_SOURCE_DIR}/shaders/*.glsl)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}
					${OPENGL_INCLUDE}
					${CUDA_INCLUDE_DIRS}
					${CMAKE_SOURCE_DIR}/dependencies/include
					${CMAKE_SOURCE_DIR}/src/framework/public_headers
)

source_group("CUDA Header Files" FILES ${CUDA_HEADERS})
source_group("CUDA Source Files" FILES ${CUDA_SOURCES})
source_group("Shader Files" FILES ${SHADERS})

if (MSVC)
	add_definitions(/D "_CRT_SECURE_NO_WARNINGS")
endif (MSVC)

cuda_add_executable(
	gpu_simulation
	${CPP_HEADERS}
	${CPP_SOURCES}
	${CUDA_HEADERS}
	${CUDA_SOURCES}
	${SHADERS}
)

add_custom_command(TARGET gpu_simulation POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory 
	"${PROJECT_SOURCE_DIR}/dependencies/libs" $<TARGET_FILE_DIR:gpu_simulation>
)

target_link_libraries(
	gpu_simulation
	${OPENGL_LIBRARY}
	${OPENGL_gl_LIBRARY}
	${OPENGL_glu_LIBRARY}
	${CMAKE_SOURCE_DIR}/dependencies/libs/FreeImage.lib
)

set_target_properties(gpu_simulation PROPERTIES DEBUG_POSTFIX d)
