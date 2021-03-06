#ifndef ROADNETWORKGRAPH_QUADRANTEDGES_H
#define ROADNETWORKGRAPH_QUADRANTEDGES_H

#include "Defines.h"

namespace RoadNetworkGraph
{

struct QuadrantEdges
{
	EdgeIndex edges[MAX_EDGES_PER_QUADRANT];
	unsigned int lastEdgeIndex;

	QuadrantEdges() : lastEdgeIndex(0) {}

};

}

#endif