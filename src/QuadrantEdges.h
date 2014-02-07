#ifndef QUADRANTEDGES_H
#define QUADRANTEDGES_H

#pragma once

#include <Constants.h>
#include <CpuGpuCompatibility.h>

struct QuadrantEdges
{
	volatile int edges[MAX_EDGES_PER_QUADRANT];
	volatile unsigned int lastEdgeIndex;

	HOST_AND_DEVICE_CODE QuadrantEdges() : lastEdgeIndex(0) {}
	HOST_AND_DEVICE_CODE ~QuadrantEdges() {}

	inline HOST_AND_DEVICE_CODE QuadrantEdges& operator = (const QuadrantEdges& other)
	{
		lastEdgeIndex = other.lastEdgeIndex;
		for (unsigned int i = 0; i < lastEdgeIndex; i++)
		{
			edges[i] = other.edges[i];
		}
		return *this;
	}

};

#endif