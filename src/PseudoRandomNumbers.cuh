#ifndef PSEUDORANDOMNUMBERS_CUH
#define PSEUDORANDOMNUMBERS_CUH

#ifdef USE_CUDA
// *********************************************************************************
// based on: http://stackoverflow.com/questions/837955/random-number-generator-in-cuda
// *********************************************************************************
__device__ int pseudoRandomNumbers(unsigned int x, unsigned int y, unsigned int z, unsigned int w)
{   
	x += 36969 * (z & 65535) + (z >> 16);
	y += 18000 * (w & 65535) + (w >> 16);

	return (x << 16) + y;  /* 32-bit result */
}

#define RAND(__x, __y) pseudoRandomNumbers(__x, __y, context->configuration->randZ, context->configuration->randW)
#else
#include <random>
#define RAND(__x, __y) rand()
#endif

#endif