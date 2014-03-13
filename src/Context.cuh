#ifndef CONTEXT_CUH
#define CONTEXT_CUH

#pragma once

#include <CpuGpuCompatibility.h>
#include <Graph.h>
#include <ImageMap.h>
#include <Primitive.h>

struct Context
{
	Graph* graph;
	ImageMap* populationDensityMap;
	ImageMap* waterBodiesMap;
	ImageMap* blockadesMap;
	ImageMap* naturalPatternMap;
	ImageMap* radialPatternMap;
	ImageMap* rasterPatternMap;
	Primitive* primitives;
	unsigned int* pseudoRandomNumbersBuffer;

	HOST_AND_DEVICE_CODE Context() {}
	HOST_AND_DEVICE_CODE ~Context() {}

};

#endif