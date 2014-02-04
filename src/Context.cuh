#ifndef CONTEXT_CUH
#define CONTEXT_CUH

#pragma once

#include <CpuGpuCompatibility.h>
#include <Graph.h>
#include <Configuration.h>
#include <ImageMap.h>
#include <Primitive.h>

struct Context
{
	Graph* graph;
	Configuration* configuration;
	ImageMap* populationDensityMap;
	ImageMap* waterBodiesMap;
	ImageMap* blockadesMap;
	ImageMap* naturalPatternMap;
	ImageMap* radialPatternMap;
	ImageMap* rasterPatternMap;
	Primitive* primitives;

	HOST_AND_DEVICE_CODE Context() {}
	HOST_AND_DEVICE_CODE ~Context() {}

};

#endif