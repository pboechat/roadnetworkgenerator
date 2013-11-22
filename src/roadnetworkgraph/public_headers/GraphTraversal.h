#ifndef GRAPHTRAVERSAL_H
#define GRAPHTRAVERSAL_H

#include <Defines.h>
#include <Graph.h>

namespace RoadNetworkGraph
{

struct GraphTraversal
{
	virtual bool operator () (const Graph& graph, VertexIndex source, VertexIndex destination, bool highway) = 0;

};

}

#endif