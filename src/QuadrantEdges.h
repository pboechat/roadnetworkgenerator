#ifndef QUADRANTEDGES_H
#define QUADRANTEDGES_H

#pragma once

#include <Constants.h>
#include <CpuGpuCompatibility.h>
#include <Collision.h>

struct QuadrantEdges
{
	volatile int edges[MAX_EDGES_PER_QUADRANT];
	volatile unsigned int lastEdgeIndex;
	volatile unsigned int numCollisions;
	Collision collisions[MAX_NUM_COLLISIONS_PER_QUADRANT];

	HOST_AND_DEVICE_CODE QuadrantEdges() : lastEdgeIndex(0), numCollisions(0) {}
	HOST_AND_DEVICE_CODE ~QuadrantEdges() {}

};

#endif