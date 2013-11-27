#ifndef ROADNETWORKGRAPH_GRAPHTRAVERSAL_H
#define ROADNETWORKGRAPH_GRAPHTRAVERSAL_H

#include <Defines.h>
#include <Graph.h>

namespace RoadNetworkGraph
{

struct GraphTraversal
{
	virtual bool operator () (const glm::vec3& source, const glm::vec3& destination, bool highway) = 0;

};

}

#endif