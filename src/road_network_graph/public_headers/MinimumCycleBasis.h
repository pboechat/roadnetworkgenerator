#ifndef ROADNETWORKGRAPH_MINIMUMCYCLEBASIS_H
#define ROADNETWORKGRAPH_MINIMUMCYCLEBASIS_H

#include <Defines.h>
#include <Vertex.h>
#include <Edge.h>
#include <Graph.h>
#include <Primitive.h>

#include <vector_math.h>

namespace RoadNetworkGraph
{
	
//////////////////////////////////////////////////////////////////////////
void extractPrimitives(Vertex* heap, unsigned int heapSize, Primitive* primitives, unsigned int primitivesSize);

}

#endif