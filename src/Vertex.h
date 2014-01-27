#ifndef ROADNETWORKGRAPH_VERTEX_H
#define ROADNETWORKGRAPH_VERTEX_H

#include "Defines.h"

#include <vector_math.h>

namespace RoadNetworkGraph
{

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE struct Vertex
{
	VertexIndex index;
	vml_vec2 position;
	EdgeIndex ins[MAX_VERTEX_IN_CONNECTIONS];
	EdgeIndex outs[MAX_VERTEX_OUT_CONNECTIONS];
	VertexIndex adjacencies[MAX_VERTEX_IN_CONNECTIONS + MAX_VERTEX_OUT_CONNECTIONS];
	unsigned int numIns;
	unsigned int numOuts;
	unsigned int numAdjacencies;
	bool removed;

	Vertex() : removed(false), numIns(0), numOuts(0), numAdjacencies(0) {}

};

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void replaceInEdge(Vertex& vertex, EdgeIndex oldInEdgeIndex, EdgeIndex newInEdgeIndex);
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void replaceAdjacency(Vertex& vertex, VertexIndex oldAdjacentVertexIndex, VertexIndex newAdjacentVertexIndex);
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void removeInEdge(Vertex& vertex, EdgeIndex edgeIndex);
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void removeOutEdge(Vertex& vertex, EdgeIndex edgeIndex);
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void removeAdjacency(Vertex& vertex, VertexIndex adjacentVertexIndex);

}

#endif