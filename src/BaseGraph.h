#ifndef ROADNETWORKGRAPH_BASEGRAPH_H
#define ROADNETWORKGRAPH_BASEGRAPH_H

#include "Defines.h"
#include <Vertex.h>
#include <Edge.h>

namespace RoadNetworkGraph
{

DEVICE_CODE struct BaseGraph
{
	VertexIndex numVertices;
	EdgeIndex numEdges;
	Vertex* vertices;
	Edge* edges;

};

//////////////////////////////////////////////////////////////////////////
GLOBAL_CODE void initializeBaseGraph(BaseGraph* graph, Vertex* vertices, Edge* edges);
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE EdgeIndex findEdge(BaseGraph* graph, Vertex& v0, Vertex& v1);
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE EdgeIndex findEdge(BaseGraph* graph, Vertex* v0, Vertex* v1);
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE EdgeIndex findEdge(BaseGraph* graph, Vertex& v0, VertexIndex v1);
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void removeEdgeReferencesInVertices(BaseGraph* graph, VertexIndex v0, VertexIndex v1);
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void removeEdgeReferencesInVertices(BaseGraph* graph, Vertex& v0, Vertex& v1);
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void removeEdgeReferencesInVertices(BaseGraph* graph, Vertex* v0, Vertex* v1);
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void removeEdgeReferencesInVertices(BaseGraph* graph, EdgeIndex edgeIndex);

}

#endif