#ifndef ROADNETWORKGRAPH_MINIMUMCYCLEBASIS_H
#define ROADNETWORKGRAPH_MINIMUMCYCLEBASIS_H

#include "Defines.h"
#include <Vertex.h>
#include <Edge.h>
#include <Graph.h>
#include <Primitive.h>

#include <Heap.h>
#include <Array.h>

#include <vector_math.h>

namespace RoadNetworkGraph
{

//////////////////////////////////////////////////////////////////////////
void allocateExtractionBuffers(unsigned int heapBufferSize, unsigned int primitivesBufferSize, unsigned int sequenceBufferSize, unsigned int visitedBufferSize);
//////////////////////////////////////////////////////////////////////////
void freeExtractionBuffers();
//////////////////////////////////////////////////////////////////////////
void extractPrimitives(Graph* graph);

}

#endif