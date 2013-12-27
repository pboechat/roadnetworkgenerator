#include <Vertex.h>

#include <exception>

namespace RoadNetworkGraph
{

//////////////////////////////////////////////////////////////////////////
void replaceInEdge(Vertex& vertex, EdgeIndex oldInEdgeIndex, EdgeIndex newInEdgeIndex)
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
		throw std::exception("!found");
	}
}

//////////////////////////////////////////////////////////////////////////
void replaceAdjacency(Vertex& vertex, VertexIndex oldAdjacentVertexIndex, VertexIndex newAdjacentVertexIndex)
{
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
		throw std::exception("!found");
	}
}

}