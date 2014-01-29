#ifndef ROADNETWORKGRAPH_QUADRANT_CUH
#define ROADNETWORKGRAPH_QUADRANT_CUH

#include "Defines.h"

#include <Box2D.cuh>

namespace RoadNetworkGraph
{

struct Quadrant
{
	unsigned int depth;
	Box2D bounds;
	QuadrantEdgesIndex edges;

	HOST_AND_DEVICE_CODE Quadrant() : edges(-1) {}
	HOST_AND_DEVICE_CODE ~Quadrant() {}
	
	/*HOST_AND_DEVICE_CODE Quadrant& operator = (const Quadrant& other)
	{
		depth = other.depth;
		bounds = other.bounds;
		edges = other.edges;
		return *this;
	}*/
	
};

}

#endif