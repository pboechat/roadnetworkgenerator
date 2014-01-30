#ifndef PSEUDORANDOMNUMBERS_CUH
#define PSEUDORANDOMNUMBERS_CUH

#ifdef USE_CUDA
// *********************************************************************************
// based on: http://stackoverflow.com/questions/837955/random-number-generator-in-cuda
// *********************************************************************************
__device__ int pseudoRandomNumbers(unsigned int z, unsigned int w)
{   
	z = 36969 * (z & 65535) + (z >> 16);
	w = 18000 * (w & 65535) + (w >> 16);

	return (z << 16) + w;  /* 32-bit result */
}

#define RAND() pseudoRandomNumbers(context->configuration->randZ, context->configuration->randW)
#else
#include <random>
#define RAND() rand()
#endif

#endif