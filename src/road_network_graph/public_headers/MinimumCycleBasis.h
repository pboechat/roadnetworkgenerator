#ifndef ROADNETWORKGRAPH_MINIMUMCYCLEBASIS_H
#define ROADNETWORKGRAPH_MINIMUMCYCLEBASIS_H

#include "Defines.h"
#include <Vertex.h>
#include <Edge.h>
#include <Graph.h>
#include <Primitive.h>
#include <StaticHeap.h>

#include <vector_math.h>

namespace RoadNetworkGraph
{
	
//////////////////////////////////////////////////////////////////////////
void extractPrimitives(StaticHeap<Vertex>& heap, Primitive* primitives, unsigned int primitivesSize);

}

#endif