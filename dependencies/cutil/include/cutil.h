#ifndef CUTIL_H
#define CUTIL_H

#include <cuda_runtime_api.h>

#include <string>
#include <sstream>
#include <exception>

__host__ void checkCudaError(cudaError_t error, const char* file, int line)
{
#ifdef _DEBUG
	if (error != cudaSuccess)
	{
		std::stringstream stream;
		stream << cudaGetErrorString(error) << " (@" << file << ":" << line << ")";
		throw std::exception(stream.str().c_str());
	}
#endif
}

__host__ void checkCudaError(const char* file, int line)
{
	checkCudaError(cudaGetLastError(), file, line);
}

#define cudaCheckedCall(call) checkCudaError(call, __FILE__, __LINE__)
#define cudaCheckError() checkCudaError(__FILE__, __LINE__)

#endif 
