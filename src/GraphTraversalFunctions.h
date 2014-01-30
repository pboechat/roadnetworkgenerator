#ifndef GRAPHTRAVERSALFUNCTIONS_H
#define GRAPHTRAVERSALFUNCTIONS_H

#pragma once

#include <Graph.h>
#include <GraphTraversal.h>
#include <Vertex.h>
#include <Edge.h>

//////////////////////////////////////////////////////////////////////////
unsigned int getValency(Graph* graph, const Vertex& vertex);

//////////////////////////////////////////////////////////////////////////
void removeDeadEndRoads(Graph* graph)
{
	bool changed;

	do
	{
		changed = false;

		for (int i = 0; i < graph->numVertices; i++)
		{
			Vertex& vertex = graph->vertices[i];

			if (vertex.removed)
			{
				continue;
			}

			if (getValency(graph, vertex) == 1)
			{
				vertex.removed = true;
				changed = true;
			}
		}
	}
	while (changed);
}

//////////////////////////////////////////////////////////////////////////
void traverse(const Graph* graph, GraphTraversal& traversal)
{
	for (int i = 0; i < graph->numEdges; i++)
	{
		const Edge& edge = graph->edges[i];
		const Vertex& sourceVertex = graph->vertices[edge.source];
		const Vertex& destinationVertex = graph->vertices[edge.destination];

		if (destinationVertex.removed || destinationVertex.removed)
		{
			continue;
		}

		if (!traversal(sourceVertex, destinationVertex, edge))
		{
			break;
		}
	}
}

//////////////////////////////////////////////////////////////////////////
unsigned int getValency(Graph* graph, const Vertex& vertex)
{
	unsigned int valency = 0;

	for (unsigned int i = 0; i < vertex.numIns; i++)
	{
		const Edge& edge = graph->edges[vertex.ins[i]];

		// FIXME: checking invariants
		if (edge.destination != vertex.index)
		{
			THROW_EXCEPTION("edge.destination != vertex.index");
		}

		const Vertex& source = graph->vertices[edge.source];

		if (source.removed)
		{
			continue;
		}

		valency++;
	}

	for (unsigned int i = 0; i < vertex.numOuts; i++)
	{
		const Edge& edge = graph->edges[vertex.outs[i]];

		// FIXME: checking invariants
		if (edge.source != vertex.index)
		{
			THROW_EXCEPTION("edge.source != vertex.index");
		}

		const Vertex& destination = graph->vertices[edge.destination];

		if (destination.removed)
		{
			continue;
		}

		valency++;
	}

	return valency;
}

#endif