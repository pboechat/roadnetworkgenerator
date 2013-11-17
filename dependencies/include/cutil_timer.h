#ifndef CUTIMER_H
#define CUTIMER_H

#include <cuda_runtime_api.h>

#define createTimer(x) \
	cudaEvent_t startEvent_##x, stopEvent_##x; \
	cudaCheckedCall(cudaEventCreate(&startEvent_##x)); \
	cudaCheckedCall(cudaEventCreate(&stopEvent_##x)); \
	float elapsedTime_##x = 0

#define startTimer(x) \
	cudaCheckedCall(cudaEventRecord(startEvent_##x, 0))

#define stopTimer(x) \
	cudaCheckedCall(cudaEventRecord(stopEvent_##x, 0)); \
	cudaCheckedCall(cudaEventSynchronize(stopEvent_##x)); \
	float tmpElapsedTime_##x; \
	cudaCheckedCall(cudaEventElapsedTime(&tmpElapsedTime_##x, startEvent_##x, stopEvent_##x)); \
	elapsedTime_##x += tmpElapsedTime_##x

#define getTimerElapsedTime(x, y) y = elapsedTime_##x

#define destroyTimer(x) \
	cudaCheckedCall(cudaEventDestroy(startEvent_##x)); \
	cudaCheckedCall(cudaEventDestroy(stopEvent_##x))

#endif