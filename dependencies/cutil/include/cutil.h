#ifndef CUTIL_H
#define CUTIL_H

#include <cuda_runtime_api.h>
#include <curand.h>

#include <string>
#include <sstream>
#include <exception>

//////////////////////////////////////////////////////////////////////////
__host__ void checkCudaError(cudaError_t error, const char* file, int line)
{
	if (error != cudaSuccess)
	{
		std::stringstream stream;
		stream << cudaGetErrorString(error) << " (@" << file << ":" << line << ")";
		throw std::exception(stream.str().c_str());
	}
}

//////////////////////////////////////////////////////////////////////////
__host__ void checkCurandError(curandStatus_t status, const char* file, int line)
{
	if (status != CURAND_STATUS_SUCCESS) 
	{
		std::stringstream stream;
		stream << "CURAND error code: " << status << " (@" << file << ":" << line << ")";
		throw std::exception(stream.str().c_str());
	}
}

//////////////////////////////////////////////////////////////////////////
__host__ void checkCudaError(const char* file, int line)
{
	checkCudaError(cudaGetLastError(), file, line);
}

//////////////////////////////////////////////////////////////////////////
inline __device__ unsigned int lanemask_lt()
{
	unsigned int lanemask;
	asm("mov.u32 %0, %lanemask_lt;" : "=r" (lanemask));
	return lanemask;
}

#define cudaCheckedCall(call) checkCudaError(call, __FILE__, __LINE__)
#define cudaCheckError() checkCudaError(__FILE__, __LINE__)
#define curandCheckedCall(call) checkCurandError(call, __FILE__, __LINE__)

#endif 
