#ifndef ROADNETWORKGRAPH_GRAPHTRAVERSAL_H
#define ROADNETWORKGRAPH_GRAPHTRAVERSAL_H

#include "Defines.h"
#include <Graph.h>

#include <vector_math.h>

namespace RoadNetworkGraph
{

struct GraphTraversal
{
	virtual bool operator () (const vml_vec2& source, const vml_vec2& destination, bool highway) = 0;

};

}

#endif