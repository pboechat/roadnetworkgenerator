#include <Graph.cuh>
#include <GraphTraversal.h>

namespace RoadNetworkGraph
{

#ifdef USE_QUADTREE
//////////////////////////////////////////////////////////////////////////
HOST_CODE void initializeGraphOnHost(Graph* graph, float snapRadius, unsigned int maxVertices, unsigned int maxEdges, Vertex* vertices, Edge* edges, QuadTree* quadtree, unsigned int maxResultsPerQuery, EdgeIndex* queryResult)
{
	graph->numVertices = 0;
	graph->numEdges = 0;
	graph->vertices = vertices;
	graph->edges = edges;
	graph->maxVertices = maxVertices;
	graph->maxEdges = maxEdges;
	graph->maxResultsPerQuery = maxResultsPerQuery;
	graph->snapRadius = snapRadius;
	graph->quadtree = quadtree;
	graph->queryResult = queryResult;
#ifdef _DEBUG
	graph->numCollisionChecks = 0;
#endif
}
#else
//////////////////////////////////////////////////////////////////////////
HOST_CODE void initializeGraphOnHost(Graph* graph, float snapRadius, unsigned int maxVertices, unsigned int maxEdges, Vertex* vertices, Edge* edges)
{
	graph->numVertices = 0;
	graph->numEdges = 0;
	graph->vertices = vertices;
	graph->edges = edges;
	graph->maxVertices = maxVertices;
	graph->maxEdges = maxEdges;
	graph->snapRadius = snapRadius;
#ifdef _DEBUG
	graph->numCollisionChecks = 0;
#endif
}
#endif

//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getValency(Graph* graph, const Vertex& vertex);

//////////////////////////////////////////////////////////////////////////
HOST_CODE void copy(Graph* graph, BaseGraph* other)
{
	other->numEdges = graph->numEdges;
	other->numVertices = graph->numVertices;
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE void removeDeadEndRoads(Graph* graph)
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
HOST_CODE void traverse(const Graph* graph, GraphTraversal& traversal)
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
HOST_CODE unsigned int getValency(Graph* graph, const Vertex& vertex)
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

#ifdef _DEBUG
//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getAllocatedVertices(Graph* graph)
{
	return graph->maxVertices;
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getVerticesInUse(Graph* graph)
{
	return graph->numVertices;
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getAllocatedEdges(Graph* graph)
{
	return graph->maxEdges;
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getEdgesInUse(Graph* graph)
{
	return graph->numEdges;
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getMaxVertexInConnections(Graph* graph)
{
	return MAX_VERTEX_IN_CONNECTIONS;
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getMaxVertexInConnectionsInUse(Graph* graph)
{
	unsigned int maxVerticesInConnectionsInUse = 0;

	for (int i = 0; i < graph->numVertices; i++)
	{
		Vertex& vertex = graph->vertices[i];

		if (vertex.numIns > maxVerticesInConnectionsInUse)
		{
			maxVerticesInConnectionsInUse = vertex.numIns;
		}
	}

	return maxVerticesInConnectionsInUse;
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getMaxVertexOutConnections(Graph* graph)
{
	return MAX_VERTEX_OUT_CONNECTIONS;
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getMaxVertexOutConnectionsInUse(Graph* graph)
{
	unsigned int maxVerticesOutConnectionsInUse = 0;

	for (int i = 0; i < graph->numVertices; i++)
	{
		Vertex& vertex = graph->vertices[i];

		if (vertex.numOuts > maxVerticesOutConnectionsInUse)
		{
			maxVerticesOutConnectionsInUse = vertex.numOuts;
		}
	}

	return maxVerticesOutConnectionsInUse;
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getAverageVertexInConnectionsInUse(Graph* graph)
{
	unsigned int totalVerticesInConnections = 0;

	for (int i = 0; i < graph->numVertices; i++)
	{
		Vertex& vertex = graph->vertices[i];
		totalVerticesInConnections += vertex.numIns;
	}

	return totalVerticesInConnections / graph->numVertices;
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getAverageVertexOutConnectionsInUse(Graph* graph)
{
	unsigned int totalVerticesOutConnections = 0;

	for (int i = 0; i < graph->numVertices; i++)
	{
		Vertex& vertex = graph->vertices[i];
		totalVerticesOutConnections += vertex.numIns;
	}

	return totalVerticesOutConnections / graph->numVertices;
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getAllocatedMemory(Graph* graph)
{
	unsigned int verticesBufferMemory = graph->maxVertices * sizeof(Vertex);
	unsigned int edgesBufferMemory = graph->maxEdges * sizeof(Vertex);
	return (verticesBufferMemory + edgesBufferMemory);
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getMemoryInUse(Graph* graph)
{
	unsigned int verticesBufferMemoryInUse = 0;

	for (int i = 0; i < graph->numVertices; i++)
	{
		Vertex& vertex = graph->vertices[i];
		// FIXME:
		verticesBufferMemoryInUse += vertex.numIns * sizeof(EdgeIndex) + vertex.numOuts * sizeof(EdgeIndex) + sizeof(vml_vec2) + 2 * sizeof(unsigned int) + sizeof(bool);
	}

	unsigned int edgesBufferMemoryInUse = graph->numEdges * sizeof(Vertex);
	return (verticesBufferMemoryInUse + edgesBufferMemoryInUse);
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned long getNumCollisionChecks(Graph* graph)
{
	return graph->numCollisionChecks;
}
#endif

}