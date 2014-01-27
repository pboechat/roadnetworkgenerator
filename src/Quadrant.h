#ifndef ROADNETWORKGRAPH_QUADRANT_H
#define ROADNETWORKGRAPH_QUADRANT_H

#include "Defines.h"

#include <Box2D.h>

namespace RoadNetworkGraph
{

HOST_AND_DEVICE_CODE struct Quadrant
{
	unsigned int depth;
	Box2D bounds;
	QuadrantEdgesIndex edges;

	Quadrant() : edges(-1) {}

};

}

#endif