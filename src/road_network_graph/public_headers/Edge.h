#ifndef ROADNETWORKGRAPH_EDGE_H
#define ROADNETWORKGRAPH_EDGE_H

#include "Defines.h"

namespace RoadNetworkGraph
{

struct Edge
{
	bool highway;
	VertexIndex source;
	VertexIndex destination;

};

}

#endif