#ifndef ROADNETWORKGRAPH_BASEGRAPH_H
#define ROADNETWORKGRAPH_BASEGRAPH_H

#include "Defines.h"
#include <Vertex.h>
#include <Edge.h>

namespace RoadNetworkGraph
{

struct BaseGraph
{
	VertexIndex numVertices;
	EdgeIndex numEdges;
	Vertex* vertices;
	Edge* edges;

};

//////////////////////////////////////////////////////////////////////////
void initializeBaseGraph(BaseGraph* graph, Vertex* vertices, Edge* edges);
//////////////////////////////////////////////////////////////////////////
EdgeIndex findEdge(BaseGraph* graph, Vertex& v0, Vertex& v1);
//////////////////////////////////////////////////////////////////////////
EdgeIndex findEdge(BaseGraph* graph, Vertex* v0, Vertex* v1);
//////////////////////////////////////////////////////////////////////////
EdgeIndex findEdge(BaseGraph* graph, Vertex& v0, VertexIndex v1);
//////////////////////////////////////////////////////////////////////////
void removeEdgeReferencesInVertices(BaseGraph* graph, VertexIndex v0, VertexIndex v1);
//////////////////////////////////////////////////////////////////////////
void removeEdgeReferencesInVertices(BaseGraph* graph, Vertex& v0, Vertex& v1);
//////////////////////////////////////////////////////////////////////////
void removeEdgeReferencesInVertices(BaseGraph* graph, Vertex* v0, Vertex* v1);
//////////////////////////////////////////////////////////////////////////
void removeEdgeReferencesInVertices(BaseGraph* graph, EdgeIndex edgeIndex);

}

#endif