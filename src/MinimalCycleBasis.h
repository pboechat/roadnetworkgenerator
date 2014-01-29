#ifndef ROADNETWORKGRAPH_MINIMALCYCLEBASIS_H
#define ROADNETWORKGRAPH_MINIMALCYCLEBASIS_H

#include "Defines.h"
#include <Vertex.cuh>
#include <Edge.cuh>
#include <BaseGraph.cuh>
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