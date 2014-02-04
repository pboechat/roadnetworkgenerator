#ifndef BASEGRAPHFUNCTIONS_CUH
#define BASEGRAPHFUNCTIONS_CUH

#pragma once

#include <BaseGraph.h>
#include <VertexFunctions.cuh>

//////////////////////////////////////////////////////////////////////////
GLOBAL_CODE void initializeBaseGraph(BaseGraph* graph, Vertex* vertices, Edge* edges)
{
	graph->numVertices = 0;
	graph->numEdges = 0;
	graph->vertices = vertices;
	graph->edges = edges;
}

//////////////////////////////////////////////////////////////////////////
HOST_AND_DEVICE_CODE int findEdge(BaseGraph* graph, Vertex& v0, Vertex& v1)
{
	for (unsigned int i = 0; i < v0.numOuts; i++)
	{
		int edgeIndex = v0.outs[i];
		if (graph->edges[edgeIndex].destination == v1.index)
		{
			return edgeIndex;
		}
	}

	for (unsigned int i = 0; i < v0.numIns; i++)
	{
		int edgeIndex = v0.ins[i];
		if (graph->edges[edgeIndex].source == v1.index)
		{
			return edgeIndex;
		}
	}

	// FIXME: checking invariants
	THROW_EXCEPTION("edge not found");

	return -1;
}

//////////////////////////////////////////////////////////////////////////
HOST_AND_DEVICE_CODE int findEdge(BaseGraph* graph, Vertex* v0, Vertex* v1)
{
	for (unsigned int i = 0; i < v0->numOuts; i++)
	{
		int edgeIndex = v0->outs[i];
		if (graph->edges[edgeIndex].destination == v1->index)
		{
			return edgeIndex;
		}
	}

	for (unsigned int i = 0; i < v0->numIns; i++)
	{
		int edgeIndex = v0->ins[i];
		if (graph->edges[edgeIndex].source == v1->index)
		{
			return edgeIndex;
		}
	}

	// FIXME: checking invariants
	THROW_EXCEPTION("edge not found");

	return -1;
}

//////////////////////////////////////////////////////////////////////////
HOST_AND_DEVICE_CODE int findEdge(BaseGraph* graph, int vertexIndex0, int vertexIndex1)
{
	Vertex& v0 = graph->vertices[vertexIndex0];
	for (unsigned int i = 0; i < v0.numOuts; i++)
	{
		int edgeIndex = v0.outs[i];
		if (graph->edges[edgeIndex].destination == vertexIndex1)
		{
			return edgeIndex;
		}
	}

	for (unsigned int i = 0; i < v0.numIns; i++)
	{
		int edgeIndex = v0.ins[i];
		if (graph->edges[edgeIndex].source == vertexIndex1)
		{
			return edgeIndex;
		}
	}

	// FIXME: checking invariants
	THROW_EXCEPTION("edge not found");

	return -1;
}

//////////////////////////////////////////////////////////////////////////
HOST_AND_DEVICE_CODE void removeEdgeReferencesInVertices(BaseGraph* graph, int edgeIndex)
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
HOST_AND_DEVICE_CODE int removeEdgeReferencesInVertices(BaseGraph* graph, Vertex& v0, Vertex& v1)
{
	int edgeIndex = findEdge(graph, v0, v1);
	Edge& edge = graph->edges[edgeIndex];

	Vertex& sourceVertex = graph->vertices[edge.source];
	Vertex& destinationVertex = graph->vertices[edge.destination];

	removeAdjacency(sourceVertex, destinationVertex.index);
	removeAdjacency(destinationVertex, sourceVertex.index);

	removeOutEdge(sourceVertex, edgeIndex);
	removeInEdge(destinationVertex, edgeIndex);

	return edgeIndex;
}

//////////////////////////////////////////////////////////////////////////
HOST_AND_DEVICE_CODE int removeEdgeReferencesInVertices(BaseGraph* graph, Vertex* v0, Vertex* v1)
{
	int edgeIndex = findEdge(graph, v0, v1);
	Edge& edge = graph->edges[edgeIndex];

	Vertex& sourceVertex = graph->vertices[edge.source];
	Vertex& destinationVertex = graph->vertices[edge.destination];

	removeAdjacency(sourceVertex, destinationVertex.index);
	removeAdjacency(destinationVertex, sourceVertex.index);

	removeOutEdge(sourceVertex, edgeIndex);
	removeInEdge(destinationVertex, edgeIndex);

	return edgeIndex;
}

//////////////////////////////////////////////////////////////////////////
HOST_AND_DEVICE_CODE int removeEdgeReferencesInVertices(BaseGraph* graph, int v0, int v1)
{
	int edgeIndex = findEdge(graph, v0, v1);
	Edge& edge = graph->edges[edgeIndex];

	Vertex& sourceVertex = graph->vertices[edge.source];
	Vertex& destinationVertex = graph->vertices[edge.destination];

	removeAdjacency(sourceVertex, destinationVertex.index);
	removeAdjacency(destinationVertex, sourceVertex.index);

	removeOutEdge(sourceVertex, edgeIndex);
	removeInEdge(destinationVertex, edgeIndex);

	return edgeIndex;
}


#endif