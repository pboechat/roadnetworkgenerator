#ifndef ROADNETWORKGRAPH_MINIMALCYCLEBASIS_H
#define ROADNETWORKGRAPH_MINIMALCYCLEBASIS_H

#include "Defines.h"
#include <Vertex.h>
#include <Edge.h>
#include <BaseGraph.h>
#include <Primitive.h>

#include <vector_math.h>

namespace RoadNetworkGraph
{

//////////////////////////////////////////////////////////////////////////
void allocateExtractionBuffers(unsigned int heapBufferSize, unsigned int sequenceBufferSize, unsigned int visitedBufferSize);
//////////////////////////////////////////////////////////////////////////
void freeExtractionBuffers();
//////////////////////////////////////////////////////////////////////////
unsigned int extractPrimitives(BaseGraph* graph, Primitive* primitivesBuffer, unsigned int maxPrimitives);

}

#endif