#ifndef MINIMALCYCLEBASIS_H
#define MINIMALCYCLEBASIS_H

#pragma once

#include <BaseGraph.h>
#include <Primitive.h>
#include <VectorMath.h>
#include <SortedSet.h>
#include <Array.h>
#include <BaseGraphFunctions.cuh>

//////////////////////////////////////////////////////////////////////////
int* g_heapBuffer = 0;
//////////////////////////////////////////////////////////////////////////
unsigned int g_heapBufferSize = 0;
//////////////////////////////////////////////////////////////////////////
int* g_sequenceBuffer = 0;
//////////////////////////////////////////////////////////////////////////
unsigned int g_sequenceBufferSize = 0;
//////////////////////////////////////////////////////////////////////////
int* g_visitedBuffer = 0;
//////////////////////////////////////////////////////////////////////////
unsigned int g_visitedBufferSize = 0;
//////////////////////////////////////////////////////////////////////////
void extractIsolatedVertex(SortedSet<int>& heap, Array<Primitive>& primitives, Vertex* v0);
//////////////////////////////////////////////////////////////////////////
void extractFilament(BaseGraph* graph, SortedSet<int>& heap, Array<Primitive>& primitives, Vertex* v0, Vertex* v1, int edgeIndex);
//////////////////////////////////////////////////////////////////////////
void extractPrimitive(BaseGraph* graph, SortedSet<int>& heap, Array<Primitive>& primitives, Vertex* v0);
//////////////////////////////////////////////////////////////////////////
Vertex* getClockwiseMostVertex(BaseGraph* graph, Vertex* previousVertex, Vertex* currentVertex);
//////////////////////////////////////////////////////////////////////////
Vertex* getCounterclockwiseMostVertex(BaseGraph* graph, Vertex* previousVertex, Vertex* currentVertex);

#define atLeft(_v0, _v1) vml_dot_perp(_v0, _v1) > 0
#define atRight(_v0, _v1) vml_dot_perp(_v0, _v1) < 0
#define convex(_v0, _v1) vml_dot_perp(_v0, _v1) >= 0

//////////////////////////////////////////////////////////////////////////
struct VertexIndexComparer : public SortedSet<int>::Comparer
{
	virtual int operator()(const int& i0, const int& i1) const
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
struct MinXMinYComparer : public SortedSet<int>::Comparer
{
	MinXMinYComparer(BaseGraph* graph) : graph(graph) {}

	virtual int operator()(const int& i0, const int& i1) const
	{
		const Vertex& v0 = graph->vertices[i0];
		const Vertex& v1 = graph->vertices[i1];

		if (v0.getPosition().x > v1.getPosition().x)
		{
			return 1;
		}

		else if (v0.getPosition().x == v1.getPosition().x)
		{
			if (v0.getPosition().y == v1.getPosition().y)
			{
				return 0;
			}

			else if (v0.getPosition().y > v1.getPosition().y)
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
void freeExtractionBuffers()
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
void allocateExtractionBuffers(unsigned int heapBufferSize, unsigned int sequenceBufferSize, unsigned int visitedBufferSize)
{
	freeExtractionBuffers();
	g_heapBufferSize = heapBufferSize;
	g_sequenceBufferSize = sequenceBufferSize;
	g_visitedBufferSize = visitedBufferSize;
	g_heapBuffer = (int*)malloc(sizeof(int) * g_heapBufferSize);
	g_sequenceBuffer = (int*)malloc(sizeof(int) * g_sequenceBufferSize);
	g_visitedBuffer = (int*)malloc(sizeof(int) * g_visitedBufferSize);
}

//////////////////////////////////////////////////////////////////////////
unsigned int extractPrimitives(BaseGraph* graph, Primitive* primitivesBuffer, unsigned int maxPrimitives)
{
	SortedSet<int> heap(g_heapBuffer, g_heapBufferSize, MinXMinYComparer(graph));
	Array<Primitive> primitives(primitivesBuffer, maxPrimitives);

	for (int vertexIndex = 0; vertexIndex < graph->numVertices; vertexIndex++)
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
void extractIsolatedVertex(SortedSet<int>& heap, Array<Primitive>& primitives, Vertex* v0)
{
	Primitive primitive;
	primitive.type = ISOLATED_VERTEX;
	insert(primitive, v0->getPosition());
	heap.remove(v0->index);
	primitives.push(primitive);
}

//////////////////////////////////////////////////////////////////////////
void extractFilament(BaseGraph* graph, SortedSet<int>& heap, Array<Primitive>& primitives, Vertex* v0, Vertex* v1, int edgeIndex)
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
			insert(primitive, v0->getPosition());
			removeEdgeReferencesInVertices(graph, v0, v1);
			v0 = v1;

			if (v0->numAdjacencies == 1)
			{
				v1 = &graph->vertices[v0->adjacencies[0]];
			}
		}

		while (v0->numAdjacencies == 1)
		{
			insert(primitive, v0->getPosition());
			v1 = &graph->vertices[v0->adjacencies[0]];
			heap.remove(v0->index);
			removeEdgeReferencesInVertices(graph, v0, v1);
			v0 = v1;
		}

		insert(primitive, v0->getPosition());

		if (v0->numAdjacencies == 0)
		{
			heap.remove(v0->index);
		}

		primitives.push(primitive);
	}
}

//////////////////////////////////////////////////////////////////////////
void extractPrimitive(BaseGraph* graph, SortedSet<int>& heap, Array<Primitive>& primitives, Vertex* v0)
{
	//SortedSet<int> visited(g_visitedBuffer, g_visitedBufferSize, VertexIndexComparer());
	Array<int> visited(g_visitedBuffer, g_visitedBufferSize);
	Array<int> sequence(g_sequenceBuffer, g_sequenceBufferSize);
	Vertex* v1 = getClockwiseMostVertex(graph, 0, v0);
	Vertex* previousVertex = v0;
	Vertex* currentVertex = v1;

	while (currentVertex != 0 && currentVertex->index != v0->index && visited.indexOf(currentVertex->index) == -1)
	{
		int edgeIndex = findEdge(graph, previousVertex, currentVertex);
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

		insert(primitive, currentVertex->getPosition());

		for (int i = (int)visited.size() - 1; i >= 0; i--)
		{
			insert(primitive, graph->vertices[visited[i]].getPosition());
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
Vertex* getClockwiseMostVertex(BaseGraph* graph, Vertex* previousVertex, Vertex* currentVertex)
{
	if (currentVertex->numAdjacencies == 0)
	{
		return 0;
	}

	Vertex* nextVertex = 0;
	vml_vec2 currentDirection;

	if (previousVertex != 0)
	{
		currentDirection = previousVertex->getPosition() - currentVertex->getPosition();

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

	vml_vec2 nextDirection = nextVertex->getPosition() - currentVertex->getPosition();
	bool isConvex = convex(nextDirection, currentDirection);

	for (unsigned int i = 0; i < currentVertex->numAdjacencies; i++)
	{
		Vertex* adjacentVertex = &graph->vertices[currentVertex->adjacencies[i]];

		if ((previousVertex != 0 && adjacentVertex->index == previousVertex->index) || adjacentVertex->index == nextVertex->index)
		{
			continue;
		}

		vml_vec2 adjacencyDirection = adjacentVertex->getPosition() - currentVertex->getPosition();

		if (isConvex)
		{
			if (atLeft(nextDirection, adjacencyDirection) && atLeft(adjacencyDirection, currentDirection))
			{
				nextVertex = adjacentVertex;
				nextDirection = adjacencyDirection;
				isConvex = convex(nextDirection, currentDirection);
			}
		}

		else
		{
			if (atRight(currentDirection, adjacencyDirection) || atRight(adjacencyDirection, nextDirection))
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
Vertex* getCounterclockwiseMostVertex (BaseGraph* graph, Vertex* previousVertex, Vertex* currentVertex)
{
	if (currentVertex->numAdjacencies == 0)
	{
		return 0;
	}

	Vertex* nextVertex = 0;
	vml_vec2 currentDirection;

	if (previousVertex != 0)
	{
		currentDirection = previousVertex->getPosition() - currentVertex->getPosition();

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

	vml_vec2 nextDirection = nextVertex->getPosition() - currentVertex->getPosition();
	bool isConvex = convex(nextDirection, currentDirection);

	for (unsigned int i = 0; i < currentVertex->numAdjacencies; i++)
	{
		Vertex* adjacentVertex = &graph->vertices[currentVertex->adjacencies[i]];

		if ((previousVertex != 0 && adjacentVertex->index == previousVertex->index) || adjacentVertex->index == nextVertex->index)
		{
			continue;
		}

		vml_vec2 adjacencyDirection = adjacentVertex->getPosition() - currentVertex->getPosition();

		if (isConvex)
		{
			if (atLeft(currentDirection, adjacencyDirection) || atLeft(adjacencyDirection, nextDirection))
			{
				nextVertex = adjacentVertex;
				nextDirection = adjacencyDirection;
				isConvex = convex(nextDirection, currentDirection);
			}
		}

		else
		{
			if (atRight(nextDirection, adjacencyDirection) && atRight(adjacencyDirection, currentDirection))
			{
				nextVertex = adjacentVertex;
				nextDirection = adjacencyDirection;
				isConvex = convex(nextDirection, currentDirection);
			}
		}
	}

	return nextVertex;
}

#endif