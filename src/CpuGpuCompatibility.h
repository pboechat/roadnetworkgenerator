#ifndef CPUGPUCOMPATIBILITY_H
#define CPUGPUCOMPATIBILITY_H

#pragma once

//////////////////////////////////////////////////////////////////////////
// GPU/CPU COMPATIBILITY
//////////////////////////////////////////////////////////////////////////

#ifdef PARALLEL
#include <cuda_runtime_api.h>
#define HOST_CODE __host__
#define DEVICE_CODE __device__
#define HOST_VARIABLE __host__
#define DEVICE_VARIABLE __device__
#define TEXTURE_VARIABLE texture<unsigned char, 2, cudaReadModeElementType>
#define CONSTANT_VARIABLE __constant__
#define GLOBAL_CODE __global__
#define HOST_AND_DEVICE_CODE __host__ __device__
	#ifdef __CUDA_ARCH__
		#define THREAD_IDX_X threadIdx.x
		#define BLOCK_DIM_X blockDim.x
		#define THROW_EXCEPTION(__message) \
			printf("%s (@%s, %d)\n", __message, __FILE__, __LINE__); \
			asm("trap;")
		#define THROW_EXCEPTION1(__format, __arg1) \
			printf(__format, __arg1); \
			printf(" (@%s, %d)\n", __FILE__, __LINE__); \
			asm("trap;")
		#define THROW_EXCEPTION2(__format, __arg1, __arg2) \
			printf(__format, __arg1, __arg2); \
			printf(" (@%s, %d)\n", __FILE__, __LINE__); \
			asm("trap;")
		#define ATOMIC_ADD(__variable, __type, __increment) atomicAdd((__type*)&__variable, (__type)__increment)
		#define ATOMIC_EXCH(__variable, __type, __value) atomicExch((__type*)&__variable, (__type)__value)
		#define ATOMIC_MAX(__variable, __type, __value) atomicMax((__type*)&__variable, (__type)__value)
		#define THREADFENCE() __threadfence()
	#else
		#include <exception>
		#include <sstream>
		template<typename T>
		inline T atomicAddMock(T* variable, T increment)
		{
			T tmp = *variable;
			*variable += increment;
			return tmp;
		}
		template<typename T>
		inline T atomicExchMock(T* variable, T value)
		{
			T tmp = *variable;
			*variable = value;
			return tmp;
		}
		template<typename T>
		inline T atomicMaxMock(T* variable, T value)
		{
			T tmp = *variable;
			*variable = (tmp < value) ? value : tmp;
			return tmp;
		}
		#define THREAD_IDX_X 0
		#define BLOCK_DIM_X 1
		#define THROW_EXCEPTION(__message) \
			{ \
				std::stringstream stringStream; \
				stringStream << __message << " (@" << __FILE__ << ", " << __FUNCTION__ << "(..), line: " << __LINE__ << ")"; \
				throw std::exception(stringStream.str().c_str()); \
			}
		#define THROW_EXCEPTION1(__format, __arg1) \
			{ \
				char str[1024]; \
				sprintf(str, __format, __arg1); \
				std::stringstream stringStream; \
				stringStream << str << " (@" << __FILE__ << ", " << __FUNCTION__ << "(..), line: " << __LINE__ << ")"; \
				throw std::exception(stringStream.str().c_str()); \
			}
		#define THROW_EXCEPTION2(__format, __arg1, __arg2) \
			{ \
				char str[1024]; \
				sprintf(str, __format, __arg1, __arg2); \
				std::stringstream stringStream; \
				stringStream << str << " (@" << __FILE__ << ", " << __FUNCTION__ << "(..), line: " << __LINE__ << ")"; \
				throw std::exception(stringStream.str().c_str()); \
			}
		#define ATOMIC_ADD(__variable, __type, __increment) atomicAddMock((__type*)&__variable, (__type)__increment)
		#define ATOMIC_EXCH(__variable, __type, __value) atomicExchMock((__type*)&__variable, (__type)__value)
		#define ATOMIC_MAX(__variable, __type, __value) atomicMaxMock((__type*)&__variable, (__type)__value)
		#define THREADFENCE()
	#endif
#else
#include <CudaTexture2DMock.h>
#include <exception>
#include <sstream>
template<typename T>
inline T atomicAddMock(T* variable, T increment)
{
	T tmp = *variable;
	*variable += increment;
	return tmp;
}
template<typename T>
inline T atomicExchMock(T* variable, T value)
{
	T tmp = *variable;
	*variable = value;
	return tmp;
}
template<typename T>
inline T atomicMaxMock(T* variable, T value)
{
	T tmp = *variable;
	*variable = (tmp < value) ? value : tmp;
	return tmp;
}
#define HOST_CODE
#define DEVICE_CODE
#define HOST_VARIABLE
#define DEVICE_VARIABLE
#define TEXTURE_VARIABLE CudaTexture2DMock
#define CONSTANT_VARIABLE
#define GLOBAL_CODE
#define HOST_AND_DEVICE_CODE
#define THREAD_IDX_X 0
#define BLOCK_DIM_X 1
#define THROW_EXCEPTION(__message) \
	{ \
		std::stringstream stringStream; \
		stringStream << __message << " (@" << __FILE__ << ", " << __FUNCTION__ << "(..), line: " << __LINE__ << ")"; \
		throw std::exception(stringStream.str().c_str()); \
	}
#define THROW_EXCEPTION1(__format, __arg1) \
	{ \
		char str[1024]; \
		sprintf(str, __format, __arg1); \
		std::stringstream stringStream; \
		stringStream << str << " (@" << __FILE__ << ", " << __FUNCTION__ << "(..), line: " << __LINE__ << ")"; \
		throw std::exception(stringStream.str().c_str()); \
	}
#define THROW_EXCEPTION2(__format, __arg1, __arg2) \
	{ \
		char str[1024]; \
		sprintf(str, __format, __arg1, __arg2); \
		std::stringstream stringStream; \
		stringStream << str << " (@" << __FILE__ << ", " << __FUNCTION__ << "(..), line: " << __LINE__ << ")"; \
		throw std::exception(stringStream.str().c_str()); \
	}
#define ATOMIC_ADD(__variable, __type, __increment) atomicAddMock((__type*)&__variable, (__type)__increment)
#define ATOMIC_EXCH(__variable, __type, __value) atomicExchMock((__type*)&__variable, (__type)__value)
#define ATOMIC_MAX(__variable, __type, __value) atomicMaxMock((__type*)&__variable, (__type)__value)
#define THREADFENCE()
#endif
#define WARN(x) printf(x)
	
#endif