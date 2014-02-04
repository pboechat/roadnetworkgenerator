#ifndef GRAPHFUNCTIONS_CUH
#define GRAPHFUNCTIONS_CUH

#pragma once

#include <Constants.h>
#include <CpuGpuCompatibility.h>
#include <Graph.h>
#include <GraphTraversal.h>
#include <Box2D.h>
#include <Line2D.h>
#include <Circle2D.h>
#include <VectorMath.h>
#ifdef USE_QUADTREE
#include <QuadTree.h>
#include <QueryResults.h>
#include <QuadTreeFunctions.cuh>
#endif
#include <VertexFunctions.cuh>
#include <Primitive.h>

//////////////////////////////////////////////////////////////////////////
enum IntersectionType
{
	NONE,
	SOURCE,
	DESTINATION,
	EDGE
};

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE int splitEdge(Graph* graph, Edge& edge, int splitVertexIndex);
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE bool checkIntersection(Graph* graph, const Line2D& newEdgeLine, Edge& edge, int sourceIndex, vml_vec2& intersection, IntersectionType& intersectionType);

#ifdef USE_QUADTREE
//////////////////////////////////////////////////////////////////////////
GLOBAL_CODE void initializeGraphOnDevice(Graph* graph, float snapRadius, unsigned int maxVertices, unsigned int maxEdges, Vertex* vertices, Edge* edges, QuadTree* quadtree)
{
	graph->numVertices = 0;
	graph->numEdges = 0;
	graph->vertices = vertices;
	graph->edges = edges;
	graph->maxVertices = maxVertices;
	graph->maxEdges = maxEdges;
	graph->snapRadius = snapRadius;
	graph->quadtree = quadtree;
#ifdef COLLECT_STATISTICS
	graph->numCollisionChecks = 0;
#endif
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE void initializeGraphOnHost(Graph* graph, float snapRadius, unsigned int maxVertices, unsigned int maxEdges, Vertex* vertices, Edge* edges, QuadTree* quadtree)
{
	graph->numVertices = 0;
	graph->numEdges = 0;
	graph->vertices = vertices;
	graph->edges = edges;
	graph->maxVertices = maxVertices;
	graph->maxEdges = maxEdges;
	graph->snapRadius = snapRadius;
	graph->quadtree = quadtree;
#ifdef COLLECT_STATISTICS
	graph->numCollisionChecks = 0;
#endif
}
#else
//////////////////////////////////////////////////////////////////////////
GLOBAL_CODE void initializeGraphOnDevice(Graph* graph, float snapRadius, unsigned int maxVertices, unsigned int maxEdges, Vertex* vertices, Edge* edges)
{
	graph->numVertices = 0;
	graph->numEdges = 0;
	graph->vertices = vertices;
	graph->edges = edges;
	graph->maxVertices = maxVertices;
	graph->maxEdges = maxEdges;
	graph->snapRadius = snapRadius;
#ifdef COLLECT_STATISTICS
	graph->numCollisionChecks = 0;
#endif
}

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
#ifdef COLLECT_STATISTICS
	graph->numCollisionChecks = 0;
#endif
}
#endif

//////////////////////////////////////////////////////////////////////////
GLOBAL_CODE void updateNonPointerFields(Graph* graph, unsigned int numVertices, unsigned int numEdges, unsigned int maxVertices, unsigned int maxEdges
#ifdef COLLECT_STATISTICS
	, unsigned long numCollisionChecks
#endif	
)
{
	graph->numVertices = numVertices;
	graph->numEdges = numEdges;
	graph->maxVertices = maxVertices;
	graph->maxEdges = maxEdges;
#ifdef COLLECT_STATISTICS
	graph->numCollisionChecks = numCollisionChecks;
#endif
}

//////////////////////////////////////////////////////////////////////////
HOST_AND_DEVICE_CODE int createVertex(Graph* graph, const vml_vec2& position)
{
	// FIXME: checking boundaries
	if (graph->numVertices >= (int)graph->maxVertices)
	{
		THROW_EXCEPTION("max. vertices overflow");
	}

	int newVertexIndex = ATOMIC_ADD(graph->numVertices, int, 1);

	Vertex* newVertex = &graph->vertices[newVertexIndex];

	newVertex->index = newVertexIndex;
	newVertex->setPosition(position);

	return newVertexIndex;
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE bool connect(Graph* graph, int sourceVertexIndex, int destinationVertexIndex, char attr1 = 0, char attr2 = 0, char attr3 = 0, char attr4 = 0)
{
	Vertex* sourceVertex = &graph->vertices[sourceVertexIndex];

	////////////////////////////
	// CHECK THIS
	////////////////////////////

	// TODO: unidirectional graph -> get rid of Ins and Outs avoiding duplicate edges
	for (unsigned int i = 0; i < sourceVertex->numAdjacencies; i++)
	{
		if (sourceVertex->adjacencies[i] == destinationVertexIndex)
		{
			return false;
		}
	}

	////////////////////////////
	// CHECK THIS
	////////////////////////////
	
	Vertex* destinationVertex = &graph->vertices[destinationVertexIndex];

	// FIXME: checking boundaries
	if (graph->numEdges >= (int)graph->maxEdges)
	{
		THROW_EXCEPTION("max. edges overflow");
	}

	int newEdgeIndex = ATOMIC_ADD(graph->numEdges, int, 1);

	Edge& newEdge = graph->edges[newEdgeIndex];

	newEdge.index = newEdgeIndex;
	newEdge.source = sourceVertexIndex;
	newEdge.destination = destinationVertexIndex;
	newEdge.attr1 = attr1;
	newEdge.attr2 = attr2;
	newEdge.attr3 = attr3;
	newEdge.attr4 = attr4;
	newEdge.owner = -1;

	// FIXME: checking boundaries
	if (sourceVertex->numOuts >= MAX_VERTEX_OUT_CONNECTIONS)
	{
		THROW_EXCEPTION("max. vertex connections (out) overflow");
	}

	unsigned int i = ATOMIC_ADD(sourceVertex->numOuts, unsigned int, 1);

	sourceVertex->outs[i] = newEdgeIndex;

	// FIXME: checking boundaries
	if (sourceVertex->numAdjacencies >= MAX_VERTEX_ADJACENCIES)
	{
		THROW_EXCEPTION("max. vertex adjacencies overflow");
	}

	i = ATOMIC_ADD(sourceVertex->numAdjacencies, unsigned int, 1);

	sourceVertex->adjacencies[i] = destinationVertexIndex;

	// FIXME: checking boundaries
	if (destinationVertex->numIns >= MAX_VERTEX_IN_CONNECTIONS)
	{
		THROW_EXCEPTION("max. vertex connections (in) overflow");
	}

	i = ATOMIC_ADD(destinationVertex->numIns, unsigned int, 1);

	destinationVertex->ins[i] = newEdgeIndex;

	// FIXME: checking boundaries
	if (destinationVertex->numAdjacencies >= MAX_VERTEX_ADJACENCIES)
	{
		THROW_EXCEPTION("max. vertex adjacencies overflow");
	}

	i = ATOMIC_ADD(destinationVertex->numAdjacencies, unsigned int, 1);

	destinationVertex->adjacencies[i] = sourceVertexIndex;

#ifdef USE_QUADTREE
	insert(graph->quadtree, newEdgeIndex, Line2D(sourceVertex->getPosition(), destinationVertex->getPosition()));
#endif

	return true;
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE int splitEdge(Graph* graph, Edge& edge, int splitVertexIndex)
{
	Vertex* splitVertex = &graph->vertices[splitVertexIndex];
	Vertex* sourceVertex = &graph->vertices[edge.source];
	Vertex* oldDestinationVertex = &graph->vertices[edge.destination];

	int oldDestinationVertexIndex = edge.destination;
	edge.destination = splitVertexIndex;

/*#ifdef USE_QUADTREE
	////////////////////////////
	// CHECK THIS
	////////////////////////////
	remove(graph->quadtree, edge.index, Line2D(sourceVertex->getPosition(), oldDestinationVertex->getPosition()));
	////////////////////////////
	// CHECK THIS
	////////////////////////////

	insert(graph->quadtree, edge.index, Line2D(sourceVertex->getPosition(), splitVertex->getPosition()));
#endif*/

	replaceAdjacency(sourceVertex, oldDestinationVertexIndex, splitVertexIndex);

	// FIXME: checking boundaries
	if (splitVertex->numIns >= MAX_VERTEX_IN_CONNECTIONS)
	{
		THROW_EXCEPTION("max. vertex connections (in) overflow");
	}

	unsigned int i = ATOMIC_ADD(splitVertex->numIns, unsigned int, 1);

	splitVertex->ins[i] = edge.index;

	// FIXME: checking boundaries
	if (splitVertex->numAdjacencies >= MAX_VERTEX_ADJACENCIES)
	{
		THROW_EXCEPTION("max. vertex adjacencies overflow");
	}

	i = ATOMIC_ADD(splitVertex->numAdjacencies, unsigned int, 1);

	splitVertex->adjacencies[i] = edge.source;

	// FIXME: checking boundaries
	if (graph->numEdges >= (int)graph->maxEdges)
	{
		THROW_EXCEPTION("max. edges overflow");
	}

	int newEdgeIndex = ATOMIC_ADD(graph->numEdges, int, 1);

	Edge& newEdge = graph->edges[newEdgeIndex];

	newEdge.index = newEdgeIndex;
	newEdge.source = splitVertexIndex;
	newEdge.destination = oldDestinationVertexIndex;
	newEdge.attr1 = edge.attr1;
	newEdge.owner = -1;

	replaceAdjacency(oldDestinationVertex, edge.source, splitVertexIndex);
	replaceInEdge(oldDestinationVertex, edge.index, newEdgeIndex);

	// FIXME: checking boundaries
	if (splitVertex->numOuts >= MAX_VERTEX_OUT_CONNECTIONS)
	{
		THROW_EXCEPTION("max. vertex connections (out) overflow");
	}

	i = ATOMIC_ADD(splitVertex->numOuts, unsigned int, 1);

	splitVertex->outs[i] = newEdgeIndex;

	// FIXME: checking boundaries
	if (splitVertex->numAdjacencies >= MAX_VERTEX_ADJACENCIES)
	{
		THROW_EXCEPTION("max. vertex adjacencies overflow");
	}

	i = ATOMIC_ADD(splitVertex->numAdjacencies, unsigned int, 1);

	splitVertex->adjacencies[i] = oldDestinationVertexIndex;
	
#ifdef USE_QUADTREE
	insert(graph->quadtree, newEdgeIndex, Line2D(splitVertex->getPosition(), oldDestinationVertex->getPosition()));
#endif

	return newEdgeIndex;
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE bool checkIntersection(Graph* graph, const Line2D& newEdgeLine, Edge& edge, int sourceIndex, vml_vec2& intersection, IntersectionType& intersectionType)
{
	// avoid intersecting parent or sibling
	if (edge.destination == sourceIndex || edge.source == sourceIndex)
	{
		return false;
	}

	vml_vec2 sourceVertexPosition = graph->vertices[edge.source].getPosition();
	vml_vec2 destinationVertexPosition = graph->vertices[edge.destination].getPosition();
		
	Line2D edgeLine(sourceVertexPosition, destinationVertexPosition);

	intersectionType = NONE;
	if (newEdgeLine.intersects(edgeLine, intersection))
	{
		if (vml_distance(sourceVertexPosition, intersection) <= graph->snapRadius)
		{
			intersection = sourceVertexPosition;
			intersectionType = SOURCE;
		}

		else if (vml_distance(destinationVertexPosition, intersection) <= graph->snapRadius)
		{
			intersection = destinationVertexPosition;
			intersectionType = DESTINATION;
		}

		else
		{
			intersectionType = EDGE;
		}
	}

#ifdef COLLECT_STATISTICS
	ATOMIC_ADD(graph->numCollisionChecks, unsigned int, 1);
#endif

	return (intersectionType != NONE);
}

#ifdef USE_QUADTREE
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE bool addHighway(Graph* graph, int sourceIndex, const vml_vec2& direction, int& destinationIndex, vml_vec2& end)
{
	vml_vec2 start = graph->vertices[sourceIndex].getPosition();
	end = start + direction;
	Line2D newEdgeLine(start, end);

	QueryResults queryResults;
	query(graph->quadtree, newEdgeLine, &queryResults);

	bool intersected = false;
	for (unsigned int i = 0; i < queryResults.numResults; i++)
	{
		QuadrantEdges* quadrantEdges = &graph->quadtree->quadrantsEdges[queryResults.results[i]];
		bool tryAgain;
		unsigned int j = 0;
		do
		{
			tryAgain = false;
			for (; j < quadrantEdges->lastEdgeIndex; j++)
			{
				vml_vec2 intersection;
				IntersectionType intersectionType;
				Edge& edge = graph->edges[quadrantEdges->edges[j]];

				if (checkIntersection(graph, newEdgeLine, edge, sourceIndex, intersection, intersectionType))
				{
					if (ATOMIC_EXCH(edge.owner, int, THREAD_IDX_X) == -1)
					{
						end = intersection;

						if (intersectionType == SOURCE)
						{
							destinationIndex = edge.source;
							connect(graph, sourceIndex, destinationIndex, true);
						}

						else if (intersectionType == DESTINATION)
						{
							destinationIndex = edge.destination;
							connect(graph, sourceIndex, destinationIndex, true);
						}

						else if (intersectionType == EDGE)
						{
							destinationIndex = createVertex(graph, intersection);
							splitEdge(graph, edge, destinationIndex);
							if (!connect(graph, sourceIndex, destinationIndex, true))
							{
								// FIXME: checking invariants
								THROW_EXCEPTION("unexpected situation");
							}
						}

						else
						{
							// FIXME: checking invariants
							THROW_EXCEPTION("unknown intersection type");
						}

						intersected = true;
						ATOMIC_EXCH(edge.owner, int, -1);
					}
					else
					{
						tryAgain = true;
					}
					break;
				} // check intersection if
			} // quadrant edges for
		} while (tryAgain); // critical-section do-while
	}  // query quadrants for

	if (intersected)
	{
		return true;
	}
	else 
	{
		destinationIndex = createVertex(graph, end);
		connect(graph, sourceIndex, destinationIndex, true);
		return false;
	}
}
#else
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE bool addHighway(Graph* graph, int sourceIndex, const vml_vec2& direction, int& destinationIndex, vml_vec2& end)
{
	vml_vec2 start = graph->vertices[sourceIndex].getPosition();
	end = start + direction;
	Line2D newEdgeLine(start, end);

	bool intersected = false;
	bool tryAgain;
	int i = 0;
	do
	{
		tryAgain = false;
		for (; i < graph->numEdges; i++)
		{
			vml_vec2 intersection;
			IntersectionType intersectionType;
			Edge& edge = graph->edges[i];

			if (checkIntersection(graph, newEdgeLine, edge, sourceIndex, intersection, intersectionType))
			{
				if (ATOMIC_EXCH(edge.owner, int, THREAD_IDX_X) == -1)
				{
					end = intersection;

					if (intersectionType == SOURCE)
					{
						destinationIndex = edge.source;
						connect(graph, sourceIndex, destinationIndex, true);
					}

					else if (intersectionType == DESTINATION)
					{
						destinationIndex = edge.destination;
						connect(graph, sourceIndex, destinationIndex, true);
					}

					else if (intersectionType == EDGE)
					{
						destinationIndex = createVertex(graph, intersection);
						splitEdge(graph, edge, destinationIndex);
						if (!connect(graph, sourceIndex, destinationIndex, true))
						{
							// FIXME: checking invariants
							THROW_EXCEPTION("unexpected situation");
						}
					}

					else
					{
						// FIXME: checking invariants
						THROW_EXCEPTION("unknown intersection type");
					}

					intersected = true;
					ATOMIC_EXCH(edge.owner, int, -1);
				}
				else
				{
					tryAgain = true;
				}
				break;
			} // check intersection if
		} // quadrant edges for
	} while (tryAgain); // critical-section do-while

	if (intersected)
	{
		return true;
	}
	else 
	{
		destinationIndex = createVertex(graph, end);
		connect(graph, sourceIndex, destinationIndex, true);
		return false;
	}
}
#endif

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE bool addStreet(Graph* graph, int sourceIndex, const vml_vec2& direction, Primitive* bounds, int& destinationIndex, vml_vec2& end)
{
	vml_vec2 start = graph->vertices[sourceIndex].getPosition();
	end = start + direction;
	Line2D newEdgeLine(start, end);
	
	bool intersected = false;
	bool tryAgain;
	unsigned int i = 0;
	do
	{
		tryAgain = false;
		for (; i < bounds->numEdges; i++)
		{
			vml_vec2 intersection;
			IntersectionType intersectionType;
			Edge& edge = graph->edges[bounds->edges[i]];

			if (checkIntersection(graph, newEdgeLine, edge, sourceIndex, intersection, intersectionType))
			{
				if (ATOMIC_EXCH(edge.owner, int, THREAD_IDX_X) == -1)
				{
					end = intersection;

					if (intersectionType == SOURCE)
					{
						destinationIndex = edge.source;
						connect(graph, sourceIndex, destinationIndex, false);
					}

					else if (intersectionType == DESTINATION)
					{
						destinationIndex = edge.destination;
						connect(graph, sourceIndex, destinationIndex, false);
					}

					else if (intersectionType == EDGE)
					{
						destinationIndex = createVertex(graph, end);
						int newEdgeIndex = splitEdge(graph, edge, destinationIndex);
						if (!connect(graph, sourceIndex, destinationIndex, false))
						{
							// FIXME: checking invariants
							THROW_EXCEPTION("unexpected situation");
						}

						unsigned int j = ATOMIC_ADD(bounds->numEdges, unsigned int, 1);
						bounds->edges[j] = newEdgeIndex;

						j = ATOMIC_ADD(bounds->numVertices, unsigned int, 1);
						bounds->vertices[j] = end;
					}

					else
					{
						// FIXME: checking invariants
						THROW_EXCEPTION("unknown intersection type");
					}

					intersected = true;
					ATOMIC_EXCH(edge.owner, int, -1);
				}
				else
				{
					tryAgain = true;
				}
				break;
			} // check intersection if
		} // primitive edges for
	} while (tryAgain); // critical-section do-while

	if (intersected)
	{
		return true;
	}
	else
	{
		destinationIndex = createVertex(graph, end);
		connect(graph, sourceIndex, destinationIndex, false);
		return false;
	}
}

#ifdef COLLECT_STATISTICS
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
		verticesBufferMemoryInUse += vertex.numIns * sizeof(int) + vertex.numOuts * sizeof(int) + sizeof(vml_vec2) + 2 * sizeof(unsigned int) + sizeof(bool);
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

#endif