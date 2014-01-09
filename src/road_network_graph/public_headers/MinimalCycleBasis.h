#ifndef ROADNETWORKGRAPH_MINIMALCYCLEBASIS_H
#define ROADNETWORKGRAPH_MINIMALCYCLEBASIS_H

#include "Defines.h"
#include <Vertex.h>
#include <Edge.h>
#include <Graph.h>
#include <Primitive.h>
#include <Array.h>

#include <vector_math.h>

namespace RoadNetworkGraph
{

//////////////////////////////////////////////////////////////////////////
void allocateExtractionBuffers(unsigned int heapBufferSize, unsigned int primitivesBufferSize, unsigned int sequenceBufferSize, unsigned int visitedBufferSize);
//////////////////////////////////////////////////////////////////////////
void freeExtractionBuffers();
//////////////////////////////////////////////////////////////////////////
Array<Primitive>& extractPrimitives(Graph* graph);

}

#endif