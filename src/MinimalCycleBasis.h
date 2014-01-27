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
HOST_CODE void allocateExtractionBuffers(unsigned int heapBufferSize, unsigned int sequenceBufferSize, unsigned int visitedBufferSize);
//////////////////////////////////////////////////////////////////////////
HOST_CODE void freeExtractionBuffers();
//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int extractPrimitives(BaseGraph* graph, Primitive* primitivesBuffer, unsigned int maxPrimitives);

}

#endif