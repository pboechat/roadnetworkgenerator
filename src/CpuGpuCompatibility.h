#ifndef CPUGPUCOMPATIBILITY_H
#define CPUGPUCOMPATIBILITY_H

#pragma once

//////////////////////////////////////////////////////////////////////////
// GPU/CPU COMPATIBILITY
//////////////////////////////////////////////////////////////////////////

#ifdef USE_CUDA
#include <cuda_runtime_api.h>
#define HOST_CODE __host__
#define DEVICE_CODE __device__
#define GLOBAL_CODE __global__
#define HOST_AND_DEVICE_CODE __host__ __device__
#define THROW_EXCEPTION(msg)
#else
#include <exception>
#define HOST_CODE
#define DEVICE_CODE
#define GLOBAL_CODE
#define HOST_AND_DEVICE_CODE
#define THROW_EXCEPTION(msg) throw std::exception(msg)
#endif

#endif