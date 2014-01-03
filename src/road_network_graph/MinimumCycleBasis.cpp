#include <MinimumCycleBasis.h>

namespace RoadNetworkGraph
{

//////////////////////////////////////////////////////////////////////////
VertexIndex* g_heapBuffer = 0;
//////////////////////////////////////////////////////////////////////////
unsigned int g_heapBufferSize = 0;
//////////////////////////////////////////////////////////////////////////
Primitive* g_primitiveBuffer = 0;
//////////////////////////////////////////////////////////////////////////
unsigned int g_primitiveBufferSize = 0;
//////////////////////////////////////////////////////////////////////////
EdgeIndex* g_sequenceBuffer = 0;
//////////////////////////////////////////////////////////////////////////
unsigned int g_sequenceBufferSize = 0;
//////////////////////////////////////////////////////////////////////////
VertexIndex* g_visitedBuffer = 0;
//////////////////////////////////////////////////////////////////////////
unsigned int g_visitedBufferSize = 0;

//////////////////////////////////////////////////////////////////////////
void extractIsolatedVertex(Heap<VertexIndex>& heap, Array<Primitive>& primitives, Vertex& v0);
//////////////////////////////////////////////////////////////////////////
void extractFilament(Graph* graph, Heap<VertexIndex>& heap, Array<Primitive>& primitives, Vertex& v0, Vertex& v1, EdgeIndex edgeIndex);
//////////////////////////////////////////////////////////////////////////
void extractPrimitive(Graph* graph, Heap<VertexIndex>& heap, Array<Primitive>& primitives, Vertex& v0);
//////////////////////////////////////////////////////////////////////////
Vertex* getClockwiseMostVertex(Graph* graph, Vertex* previousVertex, Vertex* currentVertex);
//////////////////////////////////////////////////////////////////////////
Vertex* getCounterclockwiseMostVertex(Graph* graph, Vertex* previousVertex, Vertex* currentVertex);
//////////////////////////////////////////////////////////////////////////
bool vertexComparison(const VertexIndex& v0, const VertexIndex& v1);

//////////////////////////////////////////////////////////////////////////
void allocateExtractionBuffers(unsigned int heapBufferSize, unsigned int primitivesBufferSize, unsigned int sequenceBufferSize, unsigned int visitedBufferSize)
{
	freeExtractionBuffers();
	g_heapBufferSize = heapBufferSize;
	g_primitiveBufferSize = primitivesBufferSize;
	g_sequenceBufferSize = sequenceBufferSize;
	g_visitedBufferSize = visitedBufferSize;
	g_heapBuffer = (VertexIndex*)malloc(sizeof(VertexIndex) * g_heapBufferSize);
	g_primitiveBuffer = (Primitive*)malloc(sizeof(Primitive) * g_primitiveBufferSize);
	g_sequenceBuffer = (EdgeIndex*)malloc(sizeof(EdgeIndex) * g_sequenceBufferSize);
	g_visitedBuffer = (VertexIndex*)malloc(sizeof(VertexIndex) * g_visitedBufferSize);
}

//////////////////////////////////////////////////////////////////////////
void freeExtractionBuffers()
{
	if (g_heapBuffer != 0)
	{
		free(g_heapBuffer);
		g_heapBuffer = 0;
		g_heapBufferSize = 0;
	}

	if (g_primitiveBuffer != 0)
	{
		free(g_primitiveBuffer);
		g_primitiveBuffer = 0;
		g_primitiveBufferSize = 0;
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
void extractPrimitives(Graph* graph)
{
	Heap<VertexIndex> heap(g_heapBuffer, g_heapBufferSize, vertexComparison);
	Array<Primitive> primitives(g_primitiveBuffer, g_primitiveBufferSize);

	while (!heap.empty())
	{
		Vertex& v0 = graph->vertices[heap.peekFirst()];
		if (v0.numAdjacencies == 0)
		{
			extractIsolatedVertex(heap, primitives, v0);
		}
		else if (v0.numAdjacencies == 1)
		{
			Vertex& v1 = graph->vertices[v0.adjacencies[0]];
			extractFilament(graph, heap, primitives, v0, v1, findEdge(graph, v0, v1));
		}
		else
		{
			// filament or minimal cycle
			extractPrimitive(graph, heap, primitives, v0);
		}
	}
}

//////////////////////////////////////////////////////////////////////////
void extractIsolatedVertex(Heap<Vertex>& heap, Array<Primitive>& primitives, Vertex& v0)
{
	Primitive primitive;
	primitive.type = ISOLATED_VERTEX;
	insert(primitive, v0.position);
	heap.popFirst();
	//vertices.remove(v0);
	primitives.push(primitive);
}

//////////////////////////////////////////////////////////////////////////
void extractFilament(Graph* graph, Heap<Vertex>& heap, Array<Primitive>& primitives, Vertex& v0, Vertex& v1, EdgeIndex edgeIndex)
{
	Edge& edge = graph->edges[edgeIndex];

	// is cycle edge
	if (edge.attr2 == 1)
	{
		if (v0.numAdjacencies >= 3)
		{
			removeEdge(graph, edgeIndex);
			v0 = v1;
			if (v0.numAdjacencies == 1) 
			{
				v1 = graph->vertices[v0.adjacencies[0]];
			}
		}

		while (v0.numAdjacencies == 1)
		{
			v1 = graph->vertices[v0.adjacencies[0]];
			edgeIndex = findEdge(graph, v0, v1);
			edge = graph->edges[edgeIndex];
			if (edge.attr2 == 1)
			{
				heap.remove(v0);
				removeEdge(graph, edgeIndex);
				//vertices.remove(v0);
				v0 = v1;
			}
			else
			{
				break;
			}
		}

		if (v0.numAdjacencies == 0)
		{
			heap.remove(v0);
			//vertices.remove(v0);
		}
	}
	else
	{
		Primitive primitive;
		primitive.type = FILAMENT;

		if (v0.numAdjacencies >= 3)
		{
			insert(primitive, v0.position);
			removeEdge(graph, v0, v1);
			v0 = v1;
			if (v0.numAdjacencies == 1)
			{
				v1 = graph->vertices[v0.adjacencies[0]];
			}
		}

		while (v0.numAdjacencies == 1)
		{
			insert(primitive, v0.position);
			v1 = graph->vertices[v0.adjacencies[0]];
			heap.remove(v0);
			removeEdge(graph, v0, v1);
			//vertices.remove(v0);
			v0 = v1;
		}

		insert(primitive, v0.position);
		if (v0.numAdjacencies == 0)
		{
			heap.remove(v0);
			removeEdge(graph, v0, v1);
			//vertices.remove(v0);
		}

		primitives.push(primitive);
	}
}

//////////////////////////////////////////////////////////////////////////
void extractPrimitive(Graph* graph, Heap<Vertex>& heap, Array<Primitive>& primitives, Vertex& v0)
{
	Array<VertexIndex> visited(g_visitedBuffer, g_visitedBufferSize);
	Array<EdgeIndex> sequence(g_sequenceBuffer, g_sequenceBufferSize);

	Vertex* previousVertex = &v0;
	Vertex* currentVertex = getClockwiseMostVertex(graph, 0, &v0);
	Vertex& v1 = (*currentVertex);
	Vertex* nextVertex;

	while (currentVertex != 0 && currentVertex->index != v0.index && visited.indexOf(currentVertex->index) == -1)
	{
		EdgeIndex edgeIndex = findEdge(graph, (*previousVertex), (*currentVertex));
		sequence.push(edgeIndex);
		visited.push(currentVertex->index);

		nextVertex = getCounterclockwiseMostVertex(graph, previousVertex, currentVertex);
		previousVertex = currentVertex;
		currentVertex = nextVertex;
	}

	if (currentVertex == 0)
	{
		Vertex& v2 = graph->vertices[v0.adjacencies[0]];
		// filament found, not necessarily rooted at v0
		extractFilament(graph, heap, primitives, v0, v2, findEdge(graph, v0, v2));
	}
	else if ((*currentVertex).index == v0.index)
	{
		// minimal cycle found
		Primitive primitive;
		primitive.type = MINIMAL_CYCLE;

		for (unsigned int i = 0; i < sequence.size(); i++)
		{
			Edge& edge = graph->edges[sequence[i]];
			if (i == 0)
			{
				insert(primitive, graph->vertices[edge.source].position);
			}
			insert(primitive, graph->vertices[edge.destination].position);
			edge.attr2 = 1; // is cycle edge
		}

		EdgeIndex edgeIndex = findEdge(graph, v0, v1);
		removeEdge(graph, edgeIndex);

		if (v0.numAdjacencies == 1)
		{
			// remove the filament rooted at v0
			extractFilament(graph, heap, primitives, v0, graph->vertices[v0.adjacencies[0]], edgeIndex);
		}

		if (v1.numAdjacencies == 1)
		{
			Vertex& v2 = graph->vertices[v1.adjacencies[1]];
			// remove the filament rooted at v1
			extractFilament(graph, heap, primitives, v1, v2, findEdge(graph, v1, v2));
		}
	}
	else // currentVertex was visited earlier
	{
		// A cycle has been found, but is not guaranteed to be a minimal cycle
		// This implies v0 is part of a filament
		// Locate the starting point for the filament by traversing from v0 away from the initial v1
		while (v0.numAdjacencies == 2)
		{
			if (v0.adjacencies[0] != v1.index)
			{
				v1 = v0;
				v0 = graph->vertices[v0.adjacencies[0]];
			}
			else
			{
				v1 = v0;
				v0 = graph->vertices[v0.adjacencies[1]];
			}
		}
		extractFilament(graph, heap, primitives, v0, v1, findEdge(graph, v0, v1));
	}
}

//////////////////////////////////////////////////////////////////////////
Vertex* getClockwiseMostVertex(Graph* graph, Vertex* previousVertex, Vertex* currentVertex)
{
	if (currentVertex->numAdjacencies == 0)
	{
		return 0;
	}

	vml_vec2 currentDirection;
	if (previousVertex != 0)
	{
		currentDirection = currentVertex->position - previousVertex->position;
	}
	else
	{
		currentDirection = vml_vec2(0.0f, -1.0f);
	}

	Vertex* nextVertex = 0;
	for (unsigned int i = 0; i < currentVertex->numAdjacencies; i++)
	{
		Vertex* adjacentVertex = &graph->vertices[currentVertex->adjacencies[i]];
		if (adjacentVertex->index != previousVertex->index)
		{
			nextVertex = adjacentVertex;
			break;
		}
	} 

	if (nextVertex == 0)
	{
		// FIXME: checking invariants
		throw std::exception("nextVertex == 0");
	}

	vml_vec2 nextDirection = nextVertex->position - currentVertex->position;
	
	bool convex = vml_dot_perp(nextDirection, currentDirection) != 0;
	for (unsigned int i = 0; i < currentVertex->numAdjacencies; i++)
	{
		Vertex* adjacentVertex = &graph->vertices[currentVertex->adjacencies[i]];
		vml_vec2 adjacencyDirection = adjacentVertex->position - currentVertex->position;
		if (convex)
		{
			if (vml_dot_perp(currentDirection, adjacencyDirection) < 0 || vml_dot_perp(nextDirection, adjacencyDirection) < 0)
			{
				nextVertex = adjacentVertex;
				nextDirection = adjacencyDirection;
				convex = vml_dot_perp(nextDirection, currentDirection) != 0;
			}
		}
		else
		{
			if (vml_dot_perp(currentDirection, adjacencyDirection) < 0 && vml_dot_perp(nextDirection, adjacencyDirection) < 0)
			{
				nextVertex = adjacentVertex;
				nextDirection = adjacencyDirection;
				convex = vml_dot_perp(nextDirection, currentDirection) != 0;
			}
		}
	}

	return nextVertex;
}

//////////////////////////////////////////////////////////////////////////
Vertex* getCounterclockwiseMostVertex (Graph* graph, Vertex* previousVertex, Vertex* currentVertex)
{
	if (currentVertex->numAdjacencies == 0) 
	{
		return 0;
	}

	vml_vec2 currentDirection = currentVertex->position - previousVertex->position;

	Vertex* nextVertex = 0;
	for (unsigned int i = 0; i < currentVertex->numAdjacencies; i++)
	{
		Vertex* adjacentVertex = &graph->vertices[currentVertex->adjacencies[i]];
		if (adjacentVertex->index != previousVertex->index)
		{
			nextVertex = adjacentVertex;
			break;
		}
	} 

	if (nextVertex == 0)
	{
		// FIXME: checking invariants
		throw std::exception("nextVertex == 0");
	}

	vml_vec2 nextDirection = nextVertex->position - currentVertex->position;

	bool convex = vml_dot_perp(nextDirection, currentDirection) != 0;
	for (unsigned int i = 0; i < currentVertex->numAdjacencies; i++)
	{
		Vertex* adjacentVertex = &graph->vertices[currentVertex->adjacencies[i]];
		vml_vec2 adjacencyDirection = adjacentVertex->position - currentVertex->position;
		if (convex)
		{
			if (vml_dot_perp(currentDirection, adjacencyDirection) > 0 && vml_dot_perp(nextDirection, adjacencyDirection) > 0)
			{
				nextVertex = adjacentVertex;
				nextDirection = adjacencyDirection;
				convex = vml_dot_perp(nextDirection, currentDirection) != 0;
			}
		}
		else
		{
			if (vml_dot_perp(currentDirection, adjacencyDirection) > 0 || vml_dot_perp(nextDirection, adjacencyDirection) > 0)
			{
				nextVertex = adjacentVertex;
				nextDirection = adjacencyDirection;
				convex = vml_dot_perp(nextDirection, currentDirection) != 0;
			}
		}
	}

	return nextVertex;
}

//////////////////////////////////////////////////////////////////////////
bool vertexComparison(const VertexIndex& v0, const VertexIndex& v1)
{
	// TODO:
	return false;
}

}