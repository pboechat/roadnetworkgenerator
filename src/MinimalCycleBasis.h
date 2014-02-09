#ifndef MINIMALCYCLEBASIS_H
#define MINIMALCYCLEBASIS_H

#pragma once

#include <BaseGraph.h>
#include <Primitive.h>
#include <VectorMath.h>
#include <Array.h>
#include <BaseGraphFunctions.cuh>
#include <algorithm>
#include <list>
#include <set>
#include <vector>

//////////////////////////////////////////////////////////////////////////
void extractIsolatedVertex(std::list<int>& heap, Array<Primitive>& primitives, Vertex* v0);
//////////////////////////////////////////////////////////////////////////
void extractFilament(BaseGraph* graph, std::list<int>& heap, Array<Primitive>& primitives, Vertex* v0, Vertex* v1, int edgeIndex);
//////////////////////////////////////////////////////////////////////////
void extractPrimitive(BaseGraph* graph, std::list<int>& heap, Array<Primitive>& primitives, Vertex* v0);
//////////////////////////////////////////////////////////////////////////
Vertex* getClockwiseMostVertex(BaseGraph* graph, Vertex* previousVertex, Vertex* currentVertex);
//////////////////////////////////////////////////////////////////////////
Vertex* getCounterclockwiseMostVertex(BaseGraph* graph, Vertex* previousVertex, Vertex* currentVertex);

#define atLeft(_v0, _v1) vml_dot_perp(_v0, _v1) > 0
#define atRight(_v0, _v1) vml_dot_perp(_v0, _v1) < 0
#define convex(_v0, _v1) vml_dot_perp(_v0, _v1) >= 0

//////////////////////////////////////////////////////////////////////////
struct MinXMinYComparer
{
	MinXMinYComparer(BaseGraph* graph) : graph(graph) {}

	bool operator()(int i0, int i1)
	{
		const Vertex& v0 = graph->vertices[i0];
		const Vertex& v1 = graph->vertices[i1];

		if (v0.getPosition().x > v1.getPosition().x)
		{
			return false;
		}

		else if (v0.getPosition().x == v1.getPosition().x)
		{
			if (v0.getPosition().y > v1.getPosition().y)
			{
				return false;
			}

			else if (v0.getPosition().y < v1.getPosition().y)
			{
				return true;
			}
			else
			{
				return i0 < i1;
			}
		}

		else
		{
			return true;
		}
	}

private:
	BaseGraph* graph;

};

//////////////////////////////////////////////////////////////////////////
unsigned int extractPrimitives(BaseGraph* graph, Primitive* primitivesBuffer, unsigned int maxPrimitives)
{
	std::list<int> heap;
	Array<Primitive> primitives(primitivesBuffer, maxPrimitives);

	for (int vertexIndex = 0; vertexIndex < graph->numVertices; vertexIndex++)
	{
		heap.push_back(vertexIndex);
	}

	heap.sort(MinXMinYComparer(graph));

	while (heap.size() > 0)
	{
		int i0 = heap.front();
		Vertex* v0 = &graph->vertices[i0];

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
void extractIsolatedVertex(std::list<int>& heap, Array<Primitive>& primitives, Vertex* v0)
{
	Primitive primitive;
	primitive.type = ISOLATED_VERTEX;
	insertVertex(primitive, v0->getPosition());
	heap.remove(v0->index);
	primitives.push(primitive);
}

//////////////////////////////////////////////////////////////////////////
void extractFilament(BaseGraph* graph, std::list<int>& heap, Array<Primitive>& primitives, Vertex* v0, Vertex* v1, int edgeIndex)
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
			insertVertex(primitive, v0->getPosition());
			int edgeIndex = removeEdgeReferencesInVertices(graph, v0, v1);
			insertEdge(primitive, edgeIndex);
			v0 = v1;

			if (v0->numAdjacencies == 1)
			{
				v1 = &graph->vertices[v0->adjacencies[0]];
			}
		}

		while (v0->numAdjacencies == 1)
		{
			insertVertex(primitive, v0->getPosition());
			v1 = &graph->vertices[v0->adjacencies[0]];
			heap.remove(v0->index);
			int edgeIndex = removeEdgeReferencesInVertices(graph, v0, v1);
			insertEdge(primitive, edgeIndex);
			v0 = v1;
		}

		insertVertex(primitive, v0->getPosition());

		if (v0->numAdjacencies == 0)
		{
			heap.remove(v0->index);
		}

		primitives.push(primitive);
	}
}

//////////////////////////////////////////////////////////////////////////
void extractPrimitive(BaseGraph* graph, std::list<int>& heap, Array<Primitive>& primitives, Vertex* v0)
{
	std::vector<int> visited;
	std::vector<int> sequence;
	Vertex* v1 = getClockwiseMostVertex(graph, 0, v0);
	Vertex* previousVertex = v0;
	Vertex* currentVertex = v1;

	while (currentVertex != 0 && currentVertex->index != v0->index && std::find(visited.begin(), visited.end(), currentVertex->index) == visited.end())
	{
		int edgeIndex = findEdge(graph, previousVertex, currentVertex);
		sequence.push_back(edgeIndex);
		visited.push_back(currentVertex->index);
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
		sequence.push_back(findEdge(graph, previousVertex, currentVertex));

		for (unsigned int i = 0; i < sequence.size(); i++)
		{
			int edgeIndex = sequence[i];
			graph->edges[edgeIndex].attr2 = 1; // is cycle edge
			insertEdge(primitive, edgeIndex);
		}

		insertVertex(primitive, currentVertex->getPosition());
		std::vector<int>::reverse_iterator it = visited.rbegin();
		while (it != visited.rend())
		{
			insertVertex(primitive, graph->vertices[*it].getPosition());
			it++;
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