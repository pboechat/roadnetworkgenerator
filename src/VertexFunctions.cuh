#ifndef VERTEXFUNCTIONS_CUH
#define VERTEXFUNCTIONS_CUH

#pragma once

#include <Vertex.h>

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void replaceInEdge(Vertex& vertex, int oldInEdgeIndex, int newInEdgeIndex)
{
	bool found = false;

	for (unsigned int i = 0; i < vertex.numIns; i++)
	{
		if (vertex.ins[i] == oldInEdgeIndex)
		{
			vertex.ins[i] = newInEdgeIndex;
			found = true;
			break;
		}
	}

	// FIXME: checking invariants
	if (!found)
	{
		THROW_EXCEPTION("!found");
	}
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void replaceOutEdge(Vertex& vertex, int oldOutEdgeIndex, int newOutEdgeIndex)
{
	bool found = false;

	for (unsigned int i = 0; i < vertex.numOuts; i++)
	{
		if (vertex.outs[i] == oldOutEdgeIndex)
		{
			vertex.ins[i] = newOutEdgeIndex;
			found = true;
			break;
		}
	}

	// FIXME: checking invariants
	if (!found)
	{
		THROW_EXCEPTION("!found");
	}
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void replaceAdjacency(Vertex& vertex, int oldAdjacentVertexIndex, int newAdjacentVertexIndex)
{
	// FIXME:
	for (unsigned int i = 0; i < vertex.numAdjacencies; i++)
	{
		if (vertex.adjacencies[i] == newAdjacentVertexIndex)
		{
			THROW_EXCEPTION("duplicate adjacency");
		}
	}

	bool found = false;

	for (unsigned int i = 0; i < vertex.numAdjacencies; i++)
	{
		if (vertex.adjacencies[i] == oldAdjacentVertexIndex)
		{
			vertex.adjacencies[i] = newAdjacentVertexIndex;
			found = true;
			break;
		}
	}

	// FIXME: checking invariants
	if (!found)
	{
		THROW_EXCEPTION("!found");
	}
}

//////////////////////////////////////////////////////////////////////////
HOST_AND_DEVICE_CODE void removeInEdge(Vertex& vertex, int edgeIndex)
{
	bool found = false;
	unsigned int inEdgeIndex;

	for (unsigned int i = 0; i < vertex.numIns; i++)
	{
		if (vertex.ins[i] == edgeIndex)
		{
			found = true;
			inEdgeIndex = i;
			break;
		}
	}

	// FIXME: checking invariants
	if (!found)
	{
		THROW_EXCEPTION("!found");
	}

	for (unsigned int i = inEdgeIndex; i < vertex.numIns - 1; i++)
	{
		vertex.ins[i] = vertex.ins[i + 1];
	}

	vertex.numIns--;
}

//////////////////////////////////////////////////////////////////////////
HOST_AND_DEVICE_CODE void removeOutEdge(Vertex& vertex, int edgeIndex)
{
	bool found = false;
	unsigned int outEdgeIndex;

	for (unsigned int i = 0; i < vertex.numOuts; i++)
	{
		if (vertex.outs[i] == edgeIndex)
		{
			found = true;
			outEdgeIndex = i;
			break;
		}
	}

	// FIXME: checking invariants
	if (!found)
	{
		THROW_EXCEPTION("!found");
	}

	for (unsigned int i = outEdgeIndex; i < vertex.numOuts - 1; i++)
	{
		vertex.outs[i] = vertex.outs[i + 1];
	}

	vertex.numOuts--;
}

//////////////////////////////////////////////////////////////////////////
HOST_AND_DEVICE_CODE void removeAdjacency(Vertex& vertex, int adjacentVertexIndex)
{
	bool found = false;
	unsigned int adjacencyIndex;

	for (unsigned int i = 0; i < vertex.numAdjacencies; i++)
	{
		if (vertex.adjacencies[i] == adjacentVertexIndex)
		{
			found = true;
			adjacencyIndex = i;
			break;
		}
	}

	// FIXME: checking invariants
	if (!found)
	{
		THROW_EXCEPTION("!found");
	}

	for (unsigned int i = adjacencyIndex; i < vertex.numAdjacencies - 1; i++)
	{
		vertex.adjacencies[i] = vertex.adjacencies[i + 1];
	}

	vertex.numAdjacencies--;
}

#endif