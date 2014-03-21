#ifndef QUADRANTEDGES_H
#define QUADRANTEDGES_H

#pragma once

#include <Constants.h>
#include <CpuGpuCompatibility.h>

struct QuadrantEdges
{
	/*volatile */ int edges[MAX_EDGES_PER_QUADRANT];
	volatile unsigned int lastEdgeIndex;

	HOST_AND_DEVICE_CODE QuadrantEdges() : lastEdgeIndex(0) {}
	HOST_AND_DEVICE_CODE ~QuadrantEdges() {}

};

#endif