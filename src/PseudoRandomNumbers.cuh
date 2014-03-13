#ifndef PSEUDORANDOMNUMBERS_CUH
#define PSEUDORANDOMNUMBERS_CUH

#include <CpuGpuCompatibility.h>
#include <MathExtras.h>

DEVICE_CODE unsigned int getPseudoRandomNumber(unsigned int x, unsigned int y, unsigned int w, unsigned int h, unsigned int* pseudoRandomNumbersBuffer)
{
	x = MathExtras::clamp(0u, w, x);
	y = MathExtras::clamp(0u, h, y);
	return pseudoRandomNumbersBuffer[y * w + x];
}

#define RAND(__x, __y) getPseudoRandomNumber(__x, __y, g_dConfiguration.worldWidth, g_dConfiguration.worldHeight, context->pseudoRandomNumbersBuffer)

#endif