#ifndef QUADRANT_H
#define QUADRANT_H

#include <Defines.h>

#include <AABB.h>

namespace RoadNetworkGraph
{

struct Quadrant
{
	unsigned int depth;
	AABB bounds;
	QuadrantEdgesIndex edges;

	Quadrant() : edges(-1) {}

};

}

#endif