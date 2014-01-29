#ifndef ROADNETWORKGENERATIONOBSERVER_H
#define ROADNETWORKGENERATIONOBSERVER_H

#include <Graph.cuh>
#include <Primitive.h>

class RoadNetworkGraphGenerationObserver
{
public:
	virtual void update(RoadNetworkGraph::Graph* graph, unsigned int numPrimitives, RoadNetworkGraph::Primitive* primitives) = 0;

};

#endif