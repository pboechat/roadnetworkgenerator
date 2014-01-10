#ifndef ROADNETWORKGRAPH_GRAPHTRAVERSAL_H
#define ROADNETWORKGRAPH_GRAPHTRAVERSAL_H

#include "Defines.h"
#include <Graph.h>

#include <vector_math.h>

namespace RoadNetworkGraph
{

struct GraphTraversal
{
	virtual bool operator () (const Vertex& source, const Vertex& destination, const Edge& edge) = 0;

};

}

#endif