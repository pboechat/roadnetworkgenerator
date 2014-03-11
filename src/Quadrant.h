#ifndef QUADRANT_H
#define QUADRANT_H

#pragma once

#include <CpuGpuCompatibility.h>
#include <Box2D.h>

struct Quadrant
{
	unsigned int depth;
	Box2D bounds;
	int edges;
	bool hasEdges;

	HOST_AND_DEVICE_CODE Quadrant() : edges(-1), hasEdges(false) {}
	HOST_AND_DEVICE_CODE ~Quadrant() {}
	
};

#endif