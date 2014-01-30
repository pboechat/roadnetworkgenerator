#ifndef QUADRANTEDGES_H
#define QUADRANTEDGES_H

#pragma once

#include <Constants.h>
#include <CpuGpuCompatibility.h>

struct QuadrantEdges
{
	int edges[MAX_EDGES_PER_QUADRANT];
	unsigned int lastEdgeIndex;

	HOST_AND_DEVICE_CODE QuadrantEdges() : lastEdgeIndex(0) {}
	HOST_AND_DEVICE_CODE ~QuadrantEdges() {}
	
	/*HOST_AND_DEVICE_CODE QuadrantEdges& operator = (const QuadrantEdges& other)
	{
		edges = other.edges;
		lastEdgeIndex = other.lastEdgeIndex;
		return *this;
	}*/

};

#endif