#ifndef DEFINES_H
#define DEFINES_H

#include <vector_math.h>

//////////////////////////////////////////////////////////////////////////
// GPU/CPU COMPATIBILITY
//////////////////////////////////////////////////////////////////////////

#ifdef USE_CUDA

#include <cuda_runtime_api.h>
#include <cutil.h>

#define INITIALIZE_DEVICE() \
	cudaDeviceProp deviceProperties; \
	cudaGetDeviceProperties(&deviceProperties, 0); \
	cudaSetDevice(0)
#define HOST_CODE __host__
#define DEVICE_CODE HOST_AND_DEVICE_CODE
#define GLOBAL_CODE __global__
#define HOST_AND_DEVICE_CODE __host__ HOST_AND_DEVICE_CODE
#define THROW_EXCEPTION(msg)
#define MALLOC_ON_DEVICE(__variable, __type, __amount) cudaCheckedCall(cudaMalloc((void**)&__variable, sizeof(__type) * __amount))
#define FREE_ON_DEVICE(__variable) cudaCheckedCall(cudaFree(__variable))
#define MEMCPY_HOST_TO_DEVICE(__destination, __source, __size) cudaCheckedCall(cudaMemcpy(__destination, __source, __size, cudaMemcpyHostToDevice))
#define MEMCPY_DEVICE_TO_DEVICE(__destination, __source, __size) cudaCheckedCall(cudaMemcpy(__destination, __source, __size, cudaMemcpyDeviceToDevice))
#define MEMCPY_DEVICE_TO_HOST(__destination, __source, __size) cudaCheckedCall(cudaMemcpy(__destination, __source, __size, cudaMemcpyDeviceToHost))
#define MEMSET_ON_DEVICE(__variable, __value, __size) cudaCheckedCall(cudaMemset(__variable, __value, __size))
#define INVOKE_GLOBAL_CODE(__function, __numBlocks, __numThreads) \
	__function<<<__numBlocks, __numThreads>>>(); \
	cudaCheckError()
#define INVOKE_GLOBAL_CODE1(__function, __numBlocks, __numThreads, __arg1) \
	__function<<<__numBlocks, __numThreads>>>(__arg1); \
	cudaCheckError()
#define INVOKE_GLOBAL_CODE2(__function, __numBlocks, __numThreads, __arg1, __arg2) \
	__function<<<__numBlocks, __numThreads>>>(__arg1, __arg2); \
	cudaCheckError()
#define INVOKE_GLOBAL_CODE3(__function, __numBlocks, __numThreads, __arg1, __arg2, __arg3) \
	__function<<<__numBlocks, __numThreads>>>(__arg1, __arg2, __arg3); \
	cudaCheckError()
#define INVOKE_GLOBAL_CODE4(__function, __numBlocks, __numThreads, __arg1, __arg2, __arg3, __arg4) \
	__function<<<__numBlocks, __numThreads>>>(__arg1, __arg2, __arg3, __arg4); \
	cudaCheckError()
#define INVOKE_GLOBAL_CODE5(__function, __numBlocks, __numThreads, __arg1, __arg2, __arg3, __arg4, __arg5) \
	__function<<<__numBlocks, __numThreads>>>(__arg1, __arg2, __arg3, __arg4, __arg5); \
	cudaCheckError()
#define INVOKE_GLOBAL_CODE6(__function, __numBlocks, __numThreads, __arg1, __arg2, __arg3, __arg4, __arg5, __arg6) \
	__function<<<__numBlocks, __numThreads>>>(__arg1, __arg2, __arg3, __arg4, __arg5, __arg6); \
	cudaCheckError()
#define INVOKE_GLOBAL_CODE7(__function, __numBlocks, __numThreads, __arg1, __arg2, __arg3, __arg4, __arg5, __arg6, __arg7) \
	__function<<<__numBlocks, __numThreads>>>(__arg1, __arg2, __arg3, __arg4, __arg5, __arg6, __arg7); \
	cudaCheckError()
#define INVOKE_GLOBAL_CODE8(__function, __numBlocks, __numThreads, __arg1, __arg2, __arg3, __arg4, __arg5, __arg6, __arg7, __arg8) \
	__function<<<__numBlocks, __numThreads>>>(__arg1, __arg2, __arg3, __arg4, __arg5, __arg6, __arg7, __arg8); \
	cudaCheckError()
#define INVOKE_GLOBAL_CODE9(__function, __numBlocks, __numThreads, __arg1, __arg2, __arg3, __arg4, __arg5, __arg6, __arg7, __arg8, __arg9) \
	__function<<<__numBlocks, __numThreads>>>(__arg1, __arg2, __arg3, __arg4, __arg5, __arg6, __arg7, __arg8, __arg9); \
	cudaCheckError()
#define INVOKE_GLOBAL_CODE10(__function, __numBlocks, __numThreads, __arg1, __arg2, __arg3, __arg4, __arg5, __arg6, __arg7, __arg8, __arg9, __arg10) \
	__function<<<__numBlocks, __numThreads>>>(__arg1, __arg2, __arg3, __arg4, __arg5, __arg6, __arg7, __arg8, __arg9, __arg10); \
	cudaCheckError()

#else

#include <memory>
#include <exception>

#define INITIALIZE_DEVICE()
#define HOST_CODE
#define DEVICE_CODE
#define GLOBAL_CODE
#define HOST_AND_DEVICE_CODE
#define THROW_EXCEPTION(msg) throw std::exception(msg)
#define MALLOC_ON_DEVICE(__variable, __type, __amount) \
	__variable = 0; \
	__variable = (__type*)malloc(sizeof(__type) * __amount); \
	if (__variable == 0) \
	{ \
		throw std::exception("insufficient memory"); \
	}
#define FREE_ON_DEVICE(__variable) free(__variable)
#define MEMCPY_HOST_TO_DEVICE(__destination, __source, __size) memcpy(__destination, __source, __size)
#define MEMCPY_DEVICE_TO_DEVICE(__destination, __source, __size) memcpy(__destination, __source, __size)
#define MEMCPY_DEVICE_TO_HOST(__destination, __source, __size) memcpy(__destination, __source, __size)
#define MEMSET_ON_DEVICE(__variable, __value, __size) memset(__variable, __value, __size)
#define INVOKE_GLOBAL_CODE(__function, __numBlocks, __numThreads) __function()
#define INVOKE_GLOBAL_CODE1(__function, __numBlocks, __numThreads, __arg1) __function(__arg1)
#define INVOKE_GLOBAL_CODE2(__function, __numBlocks, __numThreads, __arg1, __arg2) __function(__arg1, __arg2)
#define INVOKE_GLOBAL_CODE3(__function, __numBlocks, __numThreads, __arg1, __arg2, __arg3) __function(__arg1, __arg2, __arg3)
#define INVOKE_GLOBAL_CODE4(__function, __numBlocks, __numThreads, __arg1, __arg2, __arg3, __arg4) __function(__arg1, __arg2, __arg3, __arg4)
#define INVOKE_GLOBAL_CODE5(__function, __numBlocks, __numThreads, __arg1, __arg2, __arg3, __arg4, __arg5) __function(__arg1, __arg2, __arg3, __arg4, __arg5)
#define INVOKE_GLOBAL_CODE6(__function, __numBlocks, __numThreads, __arg1, __arg2, __arg3, __arg4, __arg5, __arg6) __function(__arg1, __arg2, __arg3, __arg4, __arg5, __arg6)
#define INVOKE_GLOBAL_CODE7(__function, __numBlocks, __numThreads, __arg1, __arg2, __arg3, __arg4, __arg5, __arg6, __arg7) __function(__arg1, __arg2, __arg3, __arg4, __arg5, __arg6, __arg7)
#define INVOKE_GLOBAL_CODE8(__function, __numBlocks, __numThreads, __arg1, __arg2, __arg3, __arg4, __arg5, __arg6, __arg7, __arg8) __function(__arg1, __arg2, __arg3, __arg4, __arg5, __arg6, __arg7, __arg8)
#define INVOKE_GLOBAL_CODE9(__function, __numBlocks, __numThreads, __arg1, __arg2, __arg3, __arg4, __arg5, __arg6, __arg7, __arg8, __arg9) __function(__arg1, __arg2, __arg3, __arg4, __arg5, __arg6, __arg7, __arg8, __arg9)
#define INVOKE_GLOBAL_CODE10(__function, __numBlocks, __numThreads, __arg1, __arg2, __arg3, __arg4, __arg5, __arg6, __arg7, __arg8, __arg9, __arg10) __function(__arg1, __arg2, __arg3, __arg4, __arg5, __arg6, __arg7, __arg8, __arg9, __arg10)

#endif

//////////////////////////////////////////////////////////////////////////
// CAMERA
//////////////////////////////////////////////////////////////////////////

#define ZNEAR 10.0f
#define ZFAR 10000.0f
#define FOVY_DEG 60.0f

//////////////////////////////////////////////////////////////////////////
// COLORS
//////////////////////////////////////////////////////////////////////////

#define VERTEX_LABEL_COLOR vml_vec4(0.0f, 1.0f, 0.0f, 1.0f)
#define EDGE_LABEL_COLOR vml_vec4(0.0f, 0.0f, 1.0f, 1.0f)
#define WHITE_COLOR vml_vec4(1.0f, 1.0f, 1.0f, 1.0f)
#define BLACK_COLOR vml_vec4(0.0f, 0.0f, 0.0f, 1.0f)
#define WATER_COLOR vml_vec4(0.5f, 0.74f, 0.98f, 1.0f)
#define GRASS_COLOR vml_vec4(0.659f, 0.8f, 0.588f, 1.0f)

//////////////////////////////////////////////////////////////////////////
// PROCEDURES
//////////////////////////////////////////////////////////////////////////

#define NUM_PROCEDURES 6

//////////////////////////////////////////////////////////////////////////
// CONFIGURATION
//////////////////////////////////////////////////////////////////////////

#define MAX_SPAWN_POINTS 100
#define MAX_CONFIGURATION_STRING_SIZE 128

//////////////////////////////////////////////////////////////////////////
// PARSING PATTERNS
//////////////////////////////////////////////////////////////////////////

#define VEC2_VECTOR_PATTERN "(\\([^\\)]+\\)\\,?)"

//////////////////////////////////////////////////////////////////////////
// GRAPH
//////////////////////////////////////////////////////////////////////////

#define MAX_VERTEX_IN_CONNECTIONS 10
#define MAX_VERTEX_OUT_CONNECTIONS 10
// MAX_VERTEX_ADJACENCIES = MAX_VERTEX_IN_CONNECTIONS + MAX_VERTEX_OUT_CONNECTIONS
#define MAX_VERTEX_ADJACENCIES 20
#define MAX_EDGES_PER_QUADRANT 1000
#define MAX_VERTICES_PER_PRIMITIVE 100

namespace RoadNetworkGraph
{

typedef int VertexIndex;
typedef int EdgeIndex;
typedef int QuadrantIndex;
typedef int QuadrantEdgesIndex;

}

//////////////////////////////////////////////////////////////////////////
//	GENERAL
//////////////////////////////////////////////////////////////////////////

#define FONT_FILE_PATH "../../../../data/fonts/arial.glf"

#ifdef _DEBUG
//////////////////////////////////////////////////////////////////////////
// DEBUG MACROS
//////////////////////////////////////////////////////////////////////////

#define toKilobytes(a) (a / 1024)
#define toMegabytes(a) (a / 1048576)
#endif

#endif