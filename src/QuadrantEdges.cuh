#ifndef ROADNETWORKGRAPH_QUADRANTEDGES_CUH
#define ROADNETWORKGRAPH_QUADRANTEDGES_CUH

#include "Defines.h"

namespace RoadNetworkGraph
{

struct QuadrantEdges
{
	EdgeIndex edges[MAX_EDGES_PER_QUADRANT];
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

}

#endif