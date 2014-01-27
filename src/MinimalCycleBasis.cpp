#include <MinimalCycleBasis.h>

#include <SortedSet.h>
#include <Array.h>

namespace RoadNetworkGraph
{

//////////////////////////////////////////////////////////////////////////
HOST_CODE VertexIndex* g_heapBuffer = 0;
//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int g_heapBufferSize = 0;
//////////////////////////////////////////////////////////////////////////
HOST_CODE EdgeIndex* g_sequenceBuffer = 0;
//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int g_sequenceBufferSize = 0;
//////////////////////////////////////////////////////////////////////////
HOST_CODE VertexIndex* g_visitedBuffer = 0;
//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int g_visitedBufferSize = 0;
//////////////////////////////////////////////////////////////////////////
HOST_CODE void extractIsolatedVertex(SortedSet<VertexIndex>& heap, Array<Primitive>& primitives, Vertex* v0);
//////////////////////////////////////////////////////////////////////////
HOST_CODE void extractFilament(BaseGraph* graph, SortedSet<VertexIndex>& heap, Array<Primitive>& primitives, Vertex* v0, Vertex* v1, EdgeIndex edgeIndex);
//////////////////////////////////////////////////////////////////////////
HOST_CODE void extractPrimitive(BaseGraph* graph, SortedSet<VertexIndex>& heap, Array<Primitive>& primitives, Vertex* v0);
//////////////////////////////////////////////////////////////////////////
HOST_CODE Vertex* getClockwiseMostVertex(BaseGraph* graph, Vertex* previousVertex, Vertex* currentVertex);
//////////////////////////////////////////////////////////////////////////
HOST_CODE Vertex* getCounterclockwiseMostVertex(BaseGraph* graph, Vertex* previousVertex, Vertex* currentVertex);

#define left(_v0, _v1) vml_dot_perp(_v0, _v1) > 0
#define right(_v0, _v1) vml_dot_perp(_v0, _v1) < 0
#define convex(_v0, _v1) vml_dot_perp(_v0, _v1) >= 0

//////////////////////////////////////////////////////////////////////////
HOST_CODE struct VertexIndexComparer : public SortedSet<VertexIndex>::Comparer
{
	virtual int operator()(const VertexIndex& i0, const VertexIndex& i1) const
	{
		if (i0 > i1)
		{
			return 1;
		}

		else if (i0 == i1)
		{
			return 0;
		}

		else
		{
			return -1;
		}
	}

};

//////////////////////////////////////////////////////////////////////////
HOST_CODE struct MinXMinYComparer : public SortedSet<VertexIndex>::Comparer
{
	MinXMinYComparer(BaseGraph* graph) : graph(graph) {}

	virtual int operator()(const VertexIndex& i0, const VertexIndex& i1) const
	{
		const Vertex& v0 = graph->vertices[i0];
		const Vertex& v1 = graph->vertices[i1];

		if (v0.position.x > v1.position.x)
		{
			return 1;
		}

		else if (v0.position.x == v1.position.x)
		{
			if (v0.position.y == v1.position.y)
			{
				return 0;
			}

			else if (v0.position.y > v1.position.y)
			{
				return 1;
			}

			else
			{
				return -1;
			}
		}

		else
		{
			return -1;
		}
	}

private:
	BaseGraph* graph;

};

//////////////////////////////////////////////////////////////////////////
HOST_CODE void allocateExtractionBuffers(unsigned int heapBufferSize, unsigned int sequenceBufferSize, unsigned int visitedBufferSize)
{
	freeExtractionBuffers();
	g_heapBufferSize = heapBufferSize;
	g_sequenceBufferSize = sequenceBufferSize;
	g_visitedBufferSize = visitedBufferSize;
	g_heapBuffer = (VertexIndex*)malloc(sizeof(VertexIndex) * g_heapBufferSize);
	g_sequenceBuffer = (EdgeIndex*)malloc(sizeof(EdgeIndex) * g_sequenceBufferSize);
	g_visitedBuffer = (VertexIndex*)malloc(sizeof(VertexIndex) * g_visitedBufferSize);
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE void freeExtractionBuffers()
{
	if (g_heapBuffer != 0)
	{
		free(g_heapBuffer);
		g_heapBuffer = 0;
		g_heapBufferSize = 0;
	}

	if (g_sequenceBuffer != 0)
	{
		free(g_sequenceBuffer);
		g_sequenceBuffer = 0;
		g_sequenceBufferSize = 0;
	}

	if (g_visitedBuffer != 0)
	{
		free(g_visitedBuffer);
		g_visitedBuffer = 0;
		g_visitedBufferSize = 0;
	}
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int extractPrimitives(BaseGraph* graph, Primitive* primitivesBuffer, unsigned int maxPrimitives)
{
	SortedSet<VertexIndex> heap(g_heapBuffer, g_heapBufferSize, MinXMinYComparer(graph));
	Array<Primitive> primitives(primitivesBuffer, maxPrimitives);

	for (VertexIndex vertexIndex = 0; vertexIndex < graph->numVertices; vertexIndex++)
	{
		heap.insert(vertexIndex);
	}

	while (heap.size() > 0)
	{
		Vertex* v0 = &graph->vertices[heap[0]];

		if (v0->numAdjacencies == 0)
		{
			extractIsolatedVertex(heap, primitives, v0);
		}

		else if (v0->numAdjacencies == 1)
		{
			Vertex* v1 = &graph->vertices[v0->adjacencies[0]];
			extractFilament(graph, heap, primitives, v0, v1, findEdge(graph, v0, v1));
		}

		else
		{
			// filament or minimal cycle
			extractPrimitive(graph, heap, primitives, v0);
		}
	}

	return primitives.size();
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE void extractIsolatedVertex(SortedSet<VertexIndex>& heap, Array<Primitive>& primitives, Vertex* v0)
{
	Primitive primitive;
	primitive.type = ISOLATED_VERTEX;
	insert(primitive, v0->position);
	heap.remove(v0->index);
	primitives.push(primitive);
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE void extractFilament(BaseGraph* graph, SortedSet<VertexIndex>& heap, Array<Primitive>& primitives, Vertex* v0, Vertex* v1, EdgeIndex edgeIndex)
{
	if (v0->numAdjacencies == 2)
	{
		// FIXME: checking invariants
		THROW_EXCEPTION("v0->numAdjacencies == 2");
	}

	// is cycle edge
	if (graph->edges[edgeIndex].attr2 == 1)
	{
		if (v0->numAdjacencies >= 3)
		{
			removeEdgeReferencesInVertices(graph, v0, v1);
			v0 = v1;

			if (v0->numAdjacencies == 1)
			{
				v1 = &graph->vertices[v0->adjacencies[0]];
			}
		}

		while (v0->numAdjacencies == 1)
		{
			v1 = &graph->vertices[v0->adjacencies[0]];
			edgeIndex = findEdge(graph, v0, v1);
			if (graph->edges[edgeIndex].attr2 == 1)
			{
				heap.remove(v0->index);
				removeEdgeReferencesInVertices(graph, edgeIndex);
				v0 = v1;
			}

			else
			{
				break;
			}
		}

		if (v0->numAdjacencies == 0)
		{
			heap.remove(v0->index);
		}
	}

	else
	{
		Primitive primitive;
		primitive.type = FILAMENT;

		if (v0->numAdjacencies >= 3)
		{
			insert(primitive, v0->position);
			removeEdgeReferencesInVertices(graph, v0, v1);
			v0 = v1;

			if (v0->numAdjacencies == 1)
			{
				v1 = &graph->vertices[v0->adjacencies[0]];
			}
		}

		while (v0->numAdjacencies == 1)
		{
			insert(primitive, v0->position);
			v1 = &graph->vertices[v0->adjacencies[0]];
			heap.remove(v0->index);
			removeEdgeReferencesInVertices(graph, v0, v1);
			v0 = v1;
		}

		insert(primitive, v0->position);

		if (v0->numAdjacencies == 0)
		{
			heap.remove(v0->index);
		}

		primitives.push(primitive);
	}
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE void extractPrimitive(BaseGraph* graph, SortedSet<VertexIndex>& heap, Array<Primitive>& primitives, Vertex* v0)
{
	//SortedSet<VertexIndex> visited(g_visitedBuffer, g_visitedBufferSize, VertexIndexComparer());
	Array<VertexIndex> visited(g_visitedBuffer, g_visitedBufferSize);
	Array<EdgeIndex> sequence(g_sequenceBuffer, g_sequenceBufferSize);
	Vertex* v1 = getClockwiseMostVertex(graph, 0, v0);
	Vertex* previousVertex = v0;
	Vertex* currentVertex = v1;

	while (currentVertex != 0 && currentVertex->index != v0->index && visited.indexOf(currentVertex->index) == -1)
	{
		EdgeIndex edgeIndex = findEdge(graph, previousVertex, currentVertex);
		sequence.push(edgeIndex);
		//visited.insert(currentVertex->index);
		visited.push(currentVertex->index);
		Vertex* nextVertex = getCounterclockwiseMostVertex(graph, previousVertex, currentVertex);
		previousVertex = currentVertex;
		currentVertex = nextVertex;
	}

	Vertex* v2;

	if (currentVertex == 0)
	{
		if (previousVertex->numAdjacencies != 1)
		{
			// FIXME: checking invariants
			THROW_EXCEPTION("previousVertex->numAdjacencies != 1");
		}

		v2 = &graph->vertices[previousVertex->adjacencies[0]];
		// filament found, not necessarily rooted at previousVertex
		extractFilament(graph, heap, primitives, previousVertex, v2, findEdge(graph, previousVertex, v2));
	}

	else if (currentVertex->index == v0->index)
	{
		// minimal cycle found
		Primitive primitive;
		primitive.type = MINIMAL_CYCLE;
		sequence.push(findEdge(graph, previousVertex, currentVertex));

		for (unsigned int i = 0; i < sequence.size(); i++)
		{
			graph->edges[sequence[i]].attr2 = 1; // is cycle edge
		}

		insert(primitive, currentVertex->position);

		for (int i = (int)visited.size() - 1; i >= 0; i--)
		{
			insert(primitive, graph->vertices[visited[i]].position);
		}

		removeEdgeReferencesInVertices(graph, v0, v1);

		if (v0->numAdjacencies == 1)
		{
			v2 = &graph->vertices[v0->adjacencies[0]];
			// remove the filament rooted at v0
			extractFilament(graph, heap, primitives, v0, v2, findEdge(graph, v0, v2));
		}

		if (v1->numAdjacencies == 1)
		{
			v2 = &graph->vertices[v1->adjacencies[0]];
			// remove the filament rooted at v1
			extractFilament(graph, heap, primitives, v1, v2, findEdge(graph, v1, v2));
		}

		primitives.push(primitive);
	}

	else // currentVertex was visited earlier
	{
		// A cycle has been found, but is not guaranteed to be a minimal cycle
		// This implies v0 is part of a filament
		// Locate the starting point for the filament by traversing from v0 away from the initial v1
		while (v0->numAdjacencies == 2)
		{
			if (v0->adjacencies[0] != v1->index)
			{
				v1 = v0;
				v0 = &graph->vertices[v0->adjacencies[0]];
			}

			else
			{
				v1 = v0;
				v0 = &graph->vertices[v0->adjacencies[1]];
			}
		}

		extractFilament(graph, heap, primitives, v0, v1, findEdge(graph, v0, v1));
	}
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE Vertex* getClockwiseMostVertex(BaseGraph* graph, Vertex* previousVertex, Vertex* currentVertex)
{
	if (currentVertex->numAdjacencies == 0)
	{
		return 0;
	}

	Vertex* nextVertex = 0;
	vml_vec2 currentDirection;

	if (previousVertex != 0)
	{
		currentDirection = previousVertex->position - currentVertex->position;

		for (unsigned int i = 0; i < currentVertex->numAdjacencies; i++)
		{
			Vertex* adjacentVertex = &graph->vertices[currentVertex->adjacencies[i]];

			if (adjacentVertex->index != previousVertex->index)
			{
				nextVertex = adjacentVertex;
				break;
			}
		}
	}

	else
	{
		currentDirection = vml_vec2(0.0f, -1.0f);
		nextVertex = &graph->vertices[currentVertex->adjacencies[0]];
	}

	if (nextVertex == 0)
	{
		return 0;
	}

	vml_vec2 nextDirection = nextVertex->position - currentVertex->position;
	bool isConvex = convex(nextDirection, currentDirection);

	for (unsigned int i = 0; i < currentVertex->numAdjacencies; i++)
	{
		Vertex* adjacentVertex = &graph->vertices[currentVertex->adjacencies[i]];

		if ((previousVertex != 0 && adjacentVertex->index == previousVertex->index) || adjacentVertex->index == nextVertex->index)
		{
			continue;
		}

		vml_vec2 adjacencyDirection = adjacentVertex->position - currentVertex->position;

		if (isConvex)
		{
			if (left(nextDirection, adjacencyDirection) && left(adjacencyDirection, currentDirection))
			{
				nextVertex = adjacentVertex;
				nextDirection = adjacencyDirection;
				isConvex = convex(nextDirection, currentDirection);
			}
		}

		else
		{
			if (right(currentDirection, adjacencyDirection) || right(adjacencyDirection, nextDirection))
			{
				nextVertex = adjacentVertex;
				nextDirection = adjacencyDirection;
				isConvex = convex(nextDirection, currentDirection);
			}
		}
	}

	return nextVertex;
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE Vertex* getCounterclockwiseMostVertex (BaseGraph* graph, Vertex* previousVertex, Vertex* currentVertex)
{
	if (currentVertex->numAdjacencies == 0)
	{
		return 0;
	}

	Vertex* nextVertex = 0;
	vml_vec2 currentDirection;

	if (previousVertex != 0)
	{
		currentDirection = previousVertex->position - currentVertex->position;

		for (unsigned int i = 0; i < currentVertex->numAdjacencies; i++)
		{
			Vertex* adjacentVertex = &graph->vertices[currentVertex->adjacencies[i]];

			if (adjacentVertex->index != previousVertex->index)
			{
				nextVertex = adjacentVertex;
				break;
			}
		}
	}

	else
	{
		currentDirection = vml_vec2(0.0f, -1.0f);
		nextVertex = &graph->vertices[currentVertex->adjacencies[0]];
	}

	if (nextVertex == 0)
	{
		return 0;
	}

	vml_vec2 nextDirection = nextVertex->position - currentVertex->position;
	bool isConvex = convex(nextDirection, currentDirection);

	for (unsigned int i = 0; i < currentVertex->numAdjacencies; i++)
	{
		Vertex* adjacentVertex = &graph->vertices[currentVertex->adjacencies[i]];

		if ((previousVertex != 0 && adjacentVertex->index == previousVertex->index) || adjacentVertex->index == nextVertex->index)
		{
			continue;
		}

		vml_vec2 adjacencyDirection = adjacentVertex->position - currentVertex->position;

		if (isConvex)
		{
			if (left(currentDirection, adjacencyDirection) || left(adjacencyDirection, nextDirection))
			{
				nextVertex = adjacentVertex;
				nextDirection = adjacencyDirection;
				isConvex = convex(nextDirection, currentDirection);
			}
		}

		else
		{
			if (right(nextDirection, adjacencyDirection) && right(adjacencyDirection, currentDirection))
			{
				nextVertex = adjacentVertex;
				nextDirection = adjacencyDirection;
				isConvex = convex(nextDirection, currentDirection);
			}
		}
	}

	return nextVertex;
}

}

