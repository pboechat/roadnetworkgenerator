#include <BaseGraph.h>

namespace RoadNetworkGraph
{

//////////////////////////////////////////////////////////////////////////
GLOBAL_CODE void initializeBaseGraph(BaseGraph* graph, Vertex* vertices, Edge* edges)
{
	graph->numVertices = 0;
	graph->numEdges = 0;
	graph->vertices = vertices;
	graph->edges = edges;
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE EdgeIndex findEdge(BaseGraph* graph, Vertex& v0, Vertex& v1)
{
	for (unsigned int i = 0; i < v0.numOuts; i++)
	{
		EdgeIndex edgeIndex = v0.outs[i];
		if (graph->edges[edgeIndex].destination == v1.index)
		{
			return edgeIndex;
		}
	}

	for (unsigned int i = 0; i < v0.numIns; i++)
	{
		EdgeIndex edgeIndex = v0.ins[i];
		if (graph->edges[edgeIndex].source == v1.index)
		{
			return edgeIndex;
		}
	}

	// FIXME: checking invariants
	THROW_EXCEPTION("edge not found");
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE EdgeIndex findEdge(BaseGraph* graph, Vertex* v0, Vertex* v1)
{
	for (unsigned int i = 0; i < v0->numOuts; i++)
	{
		EdgeIndex edgeIndex = v0->outs[i];
		if (graph->edges[edgeIndex].destination == v1->index)
		{
			return edgeIndex;
		}
	}

	for (unsigned int i = 0; i < v0->numIns; i++)
	{
		EdgeIndex edgeIndex = v0->ins[i];
		if (graph->edges[edgeIndex].source == v1->index)
		{
			return edgeIndex;
		}
	}

	// FIXME: checking invariants
	THROW_EXCEPTION("edge not found");
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE EdgeIndex findEdge(BaseGraph* graph, VertexIndex vertexIndex0, VertexIndex vertexIndex1)
{
	Vertex& v0 = graph->vertices[vertexIndex0];
	for (unsigned int i = 0; i < v0.numOuts; i++)
	{
		EdgeIndex edgeIndex = v0.outs[i];
		if (graph->edges[edgeIndex].destination == vertexIndex1)
		{
			return edgeIndex;
		}
	}

	for (unsigned int i = 0; i < v0.numIns; i++)
	{
		EdgeIndex edgeIndex = v0.ins[i];
		if (graph->edges[edgeIndex].source == vertexIndex1)
		{
			return edgeIndex;
		}
	}

	// FIXME: checking invariants
	THROW_EXCEPTION("edge not found");
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void removeEdgeReferencesInVertices(BaseGraph* graph, EdgeIndex edgeIndex)
{
	Edge& edge = graph->edges[edgeIndex];

	Vertex& sourceVertex = graph->vertices[edge.source];
	Vertex& destinationVertex = graph->vertices[edge.destination];

	removeAdjacency(sourceVertex, destinationVertex.index);
	removeAdjacency(destinationVertex, sourceVertex.index);

	removeOutEdge(sourceVertex, edgeIndex);
	removeInEdge(destinationVertex, edgeIndex);
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void removeEdgeReferencesInVertices(BaseGraph* graph, Vertex& v0, Vertex& v1)
{
	EdgeIndex edgeIndex = findEdge(graph, v0, v1);
	Edge& edge = graph->edges[edgeIndex];

	Vertex& sourceVertex = graph->vertices[edge.source];
	Vertex& destinationVertex = graph->vertices[edge.destination];

	removeAdjacency(sourceVertex, destinationVertex.index);
	removeAdjacency(destinationVertex, sourceVertex.index);

	removeOutEdge(sourceVertex, edgeIndex);
	removeInEdge(destinationVertex, edgeIndex);
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void removeEdgeReferencesInVertices(BaseGraph* graph, Vertex* v0, Vertex* v1)
{
	EdgeIndex edgeIndex = findEdge(graph, v0, v1);
	Edge& edge = graph->edges[edgeIndex];

	Vertex& sourceVertex = graph->vertices[edge.source];
	Vertex& destinationVertex = graph->vertices[edge.destination];

	removeAdjacency(sourceVertex, destinationVertex.index);
	removeAdjacency(destinationVertex, sourceVertex.index);

	removeOutEdge(sourceVertex, edgeIndex);
	removeInEdge(destinationVertex, edgeIndex);
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void removeEdgeReferencesInVertices(BaseGraph* graph, VertexIndex v0, VertexIndex v1)
{
	EdgeIndex edgeIndex = findEdge(graph, v0, v1);
	Edge& edge = graph->edges[edgeIndex];

	Vertex& sourceVertex = graph->vertices[edge.source];
	Vertex& destinationVertex = graph->vertices[edge.destination];

	removeAdjacency(sourceVertex, destinationVertex.index);
	removeAdjacency(destinationVertex, sourceVertex.index);

	removeOutEdge(sourceVertex, edgeIndex);
	removeInEdge(destinationVertex, edgeIndex);
}

}