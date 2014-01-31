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

//////////////////////////////////////////////////////////////////////////
enum IntersectionType
{
	NONE,
	SOURCE,
	DESTINATION,
	EDGE
};

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void splitEdge(Graph* graph, int edge, int vertex);
#ifdef USE_QUADTREE
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE bool checkIntersection(Graph* graph, const Line2D& newEdgeLine, QueryResults* queryResults, int source, int& edgeIndex, vml_vec2& closestIntersection, IntersectionType& intersectionType);
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE bool checkSnapping(Graph* graph, const Circle2D& snapCircle, QueryResults* queryResults, int source, vml_vec2& closestSnapping, int& edgeIndex);
#else
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE bool checkIntersection(Graph* graph, const Line2D& newEdgeLine, int source, int& edgeIndex, vml_vec2& closestIntersection, IntersectionType& intersectionType);
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE bool checkSnapping(Graph* graph, const Circle2D& snapCircle, int source, vml_vec2& closestSnapping, int& edgeIndex);
#endif

#ifdef USE_QUADTREE
//////////////////////////////////////////////////////////////////////////
GLOBAL_CODE void initializeGraphOnDevice(Graph* graph, float snapRadius, unsigned int maxVertices, unsigned int maxEdges, Vertex* vertices, Edge* edges, QuadTree* quadtree, unsigned int maxQueryResults, QueryResults* queryResults)
{
	graph->numVertices = 0;
	graph->numEdges = 0;
	graph->vertices = vertices;
	graph->edges = edges;
	graph->maxVertices = maxVertices;
	graph->maxEdges = maxEdges;
	graph->snapRadius = snapRadius;
	graph->quadtree = quadtree;
	graph->lastUsedQueryResults = 0;
	graph->maxQueryResults = maxQueryResults;
	graph->queryResults = queryResults;

	graph->owner = -1;

#ifdef _DEBUG
	graph->numCollisionChecks = 0;
#endif
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE void initializeGraphOnHost(Graph* graph, float snapRadius, unsigned int maxVertices, unsigned int maxEdges, Vertex* vertices, Edge* edges, QuadTree* quadtree, unsigned int maxQueryResults, QueryResults* queryResults)
{
	graph->numVertices = 0;
	graph->numEdges = 0;
	graph->vertices = vertices;
	graph->edges = edges;
	graph->maxVertices = maxVertices;
	graph->maxEdges = maxEdges;
	graph->snapRadius = snapRadius;
	graph->quadtree = quadtree;
	graph->lastUsedQueryResults = 0;
	graph->maxQueryResults = maxQueryResults;
	graph->queryResults = queryResults;

	graph->owner = -1;

#ifdef _DEBUG
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

	// DEBUG:
	graph->owner = -1;

#ifdef _DEBUG
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

	// DEBUG:
	graph->owner = -1;

#ifdef _DEBUG
	graph->numCollisionChecks = 0;
#endif
}
#endif

//////////////////////////////////////////////////////////////////////////
GLOBAL_CODE void updateNonPointerFields(Graph* graph, unsigned int numVertices, unsigned int numEdges, unsigned int maxVertices, unsigned int maxEdges, int owner
#ifdef USE_QUADTREE
	, unsigned int lastUsedQueryResults
	, unsigned int maxQueryResults
#endif
#ifdef _DEBUG
	, unsigned long numCollisionChecks
#endif	
)
{
	graph->numVertices = numVertices;
	graph->numEdges = numEdges;
	graph->maxVertices = maxVertices;
	graph->maxEdges = maxEdges;
	graph->owner = owner;

#ifdef USE_QUADTREE
	graph->lastUsedQueryResults = lastUsedQueryResults;
	graph->maxQueryResults = maxQueryResults;
#endif
#ifdef _DEBUG
	graph->numCollisionChecks = numCollisionChecks;
#endif
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE vml_vec2 getPosition(Graph* graph, int vertexIndex)
{
	// FIXME: checking invariants
	if (vertexIndex >= graph->numVertices)
	{
		THROW_EXCEPTION("invalid vertex index");
	}

	Vertex* vertex = &graph->vertices[vertexIndex];
	vml_vec2 position = vertex->getPosition();
	return position;
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

	// FIXME: checking boundaries
	if (graph->numVertices >= (int)graph->maxVertices)
	{
		THROW_EXCEPTION("max. vertices overflow");
	}

	Vertex* newVertex = &graph->vertices[newVertexIndex];

	newVertex->index = newVertexIndex;
	newVertex->setPosition(position);

	return newVertexIndex;
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE bool connect(Graph* graph, int sourceVertexIndex, int destinationVertexIndex, bool highway)
{
	Vertex* sourceVertex = &graph->vertices[sourceVertexIndex];

	// TODO: unidirectional graph -> get rid of Ins and Outs avoiding duplicate edges
	for (unsigned int i = 0; i < sourceVertex->numAdjacencies; i++)
	{
		if (sourceVertex->adjacencies[i] == destinationVertexIndex)
		{
			return false;
		}
	}
	
	Vertex* destinationVertex = &graph->vertices[destinationVertexIndex];

	// FIXME: checking boundaries
	if (graph->numEdges >= (int)graph->maxEdges)
	{
		THROW_EXCEPTION("max. edges overflow");
	}

	int newEdgeIndex = ATOMIC_ADD(graph->numEdges, int, 1);

	if (graph->numEdges >= (int)graph->maxEdges)
	{
		THROW_EXCEPTION("max. edges overflow");
	}

	Edge& newEdge = graph->edges[newEdgeIndex];

	newEdge.index = newEdgeIndex;
	newEdge.source = sourceVertexIndex;
	newEdge.destination = destinationVertexIndex;
	newEdge.attr1 = (highway) ? 1 : 0;

	// FIXME: checking boundaries
	if (sourceVertex->numOuts >= MAX_VERTEX_OUT_CONNECTIONS)
	{
		THROW_EXCEPTION("max. vertex connections (out) overflow");
	}

	sourceVertex->outs[sourceVertex->numOuts++] = newEdgeIndex;

	// FIXME: checking boundaries
	if (sourceVertex->numAdjacencies >= MAX_VERTEX_ADJACENCIES)
	{
		THROW_EXCEPTION("max. vertex adjacencies overflow");
	}

	sourceVertex->adjacencies[sourceVertex->numAdjacencies++] = destinationVertexIndex;

	// FIXME: checking boundaries
	if (destinationVertex->numIns >= MAX_VERTEX_IN_CONNECTIONS)
	{
		THROW_EXCEPTION("max. vertex connections (in) overflow");
	}

	destinationVertex->ins[destinationVertex->numIns++] = newEdgeIndex;

	// FIXME: checking boundaries
	if (destinationVertex->numAdjacencies >= MAX_VERTEX_ADJACENCIES)
	{
		THROW_EXCEPTION("max. vertex adjacencies overflow");
	}

	destinationVertex->adjacencies[destinationVertex->numAdjacencies++] = sourceVertexIndex;

#ifdef USE_QUADTREE
	insert(graph->quadtree, newEdgeIndex, Line2D(sourceVertex->getPosition(), destinationVertex->getPosition()));
#endif

	return true;
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void splitEdge(Graph* graph, int edgeIndex, int splitVertexIndex)
{
	Edge& edge = graph->edges[edgeIndex];

	Vertex* splitVertex = &graph->vertices[splitVertexIndex];
	Vertex* sourceVertex = &graph->vertices[edge.source];
	Vertex* oldDestinationVertex = &graph->vertices[edge.destination];

	int oldDestinationVertexIndex = edge.destination;
	edge.destination = splitVertexIndex;

#ifdef USE_QUADTREE
	remove(graph->quadtree, edgeIndex, Line2D(sourceVertex->getPosition(), oldDestinationVertex->getPosition()));
	insert(graph->quadtree, edgeIndex, Line2D(sourceVertex->getPosition(), splitVertex->getPosition()));
#endif

	replaceAdjacency(sourceVertex, oldDestinationVertexIndex, splitVertexIndex);

	// FIXME: checking boundaries
	if (splitVertex->numIns >= MAX_VERTEX_IN_CONNECTIONS)
	{
		THROW_EXCEPTION("max. vertex connections (in) overflow");
	}

	splitVertex->ins[splitVertex->numIns++] = edgeIndex;

	// FIXME: checking boundaries
	if (splitVertex->numAdjacencies >= MAX_VERTEX_ADJACENCIES)
	{
		THROW_EXCEPTION("max. vertex adjacencies overflow");
	}

	splitVertex->adjacencies[splitVertex->numAdjacencies++] = edge.source;

	// FIXME: checking boundaries
	if (graph->numEdges >= (int)graph->maxEdges)
	{
		THROW_EXCEPTION("max. edges overflow");
	}

	int newEdgeIndex = ATOMIC_ADD(graph->numEdges, int, 1);

	// FIXME: checking boundaries
	if (graph->numEdges >= (int)graph->maxEdges)
	{
		THROW_EXCEPTION("max. edges overflow");
	}

	Edge& newEdge = graph->edges[newEdgeIndex];
	newEdge.index = newEdgeIndex;
	newEdge.source = splitVertexIndex;
	newEdge.destination = oldDestinationVertexIndex;
	newEdge.attr1 = edge.attr1;

	replaceAdjacency(oldDestinationVertex, edge.source, splitVertexIndex);
	replaceInEdge(oldDestinationVertex, edgeIndex, newEdgeIndex);

	// FIXME: checking boundaries
	if (splitVertex->numOuts >= MAX_VERTEX_OUT_CONNECTIONS)
	{
		THROW_EXCEPTION("max. vertex connections (out) overflow");
	}

	splitVertex->outs[splitVertex->numOuts++] = newEdgeIndex;

	// FIXME: checking boundaries
	if (splitVertex->numAdjacencies >= MAX_VERTEX_ADJACENCIES)
	{
		THROW_EXCEPTION("max. vertex adjacencies overflow");
	}

	splitVertex->adjacencies[splitVertex->numAdjacencies++] = oldDestinationVertexIndex;
	
#ifdef USE_QUADTREE
	insert(graph->quadtree, newEdgeIndex, Line2D(splitVertex->getPosition(), oldDestinationVertex->getPosition()));
#endif
}

#ifdef USE_QUADTREE
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE bool addRoad(Graph* graph, int sourceIndex, const vml_vec2& direction, int& newVertexIndex, vml_vec2& end, bool highway)
{
	int edgeIndex;
	vml_vec2 snapping;
	vml_vec2 intersection;
	IntersectionType intersectionType;

	vml_vec2 start = getPosition(graph, sourceIndex);
	end = start + direction;
	Line2D newEdgeLine(start, end);
	Box2D newEdgeBounds(newEdgeLine);

	// critical section
	bool returnValue = false;
	bool leaveLoop = false;
	while (!leaveLoop)
	{ 
		if (ATOMIC_EXCH(graph->owner, int, THREAD_IDX_X) == -1)
		{
			int queryResultIndex = ATOMIC_ADD(graph->lastUsedQueryResults, int, 1);
			QueryResults* queryResults = &graph->queryResults[queryResultIndex % graph->maxQueryResults];
			query(graph->quadtree, newEdgeBounds, queryResults);

			if (checkIntersection(graph, newEdgeLine, queryResults, sourceIndex, edgeIndex, intersection, intersectionType))
			{
				end = intersection;

				Edge& edge = graph->edges[edgeIndex];

				if (intersectionType == SOURCE)
				{
					newVertexIndex = edge.source;
					connect(graph, sourceIndex, newVertexIndex, highway);
				}

				else if (intersectionType == DESTINATION)
				{
					newVertexIndex = edge.destination;
					connect(graph, sourceIndex, newVertexIndex, highway);
				}

				else if (intersectionType == EDGE)
				{
					newVertexIndex = createVertex(graph, end);
					splitEdge(graph, edgeIndex, newVertexIndex);
					if (!connect(graph, sourceIndex, newVertexIndex, highway))
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

				returnValue = true;
			}

			else
			{
				Circle2D snapCircle(end, graph->snapRadius);
				query(graph->quadtree, snapCircle, queryResults);

				if (checkSnapping(graph, snapCircle, queryResults, sourceIndex, snapping, edgeIndex))
				{
					end = snapping;
					newVertexIndex = createVertex(graph, end);
					splitEdge(graph, edgeIndex, newVertexIndex);
					if (!connect(graph, sourceIndex, newVertexIndex, highway))
					{
						// FIXME: checking invariants
						THROW_EXCEPTION("unexpected situation");
					}

					returnValue = true;
				}

				else
				{
					newVertexIndex = createVertex(graph, end);
					if (connect(graph, sourceIndex, newVertexIndex, highway))
					{
						returnValue = false;
					}
					else
					{
						returnValue = true;
					}
				}
			}
		
			THREADFENCE();

			leaveLoop = true;
			ATOMIC_EXCH(graph->owner, int, -1);
		}
	}

	return returnValue;
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE bool checkIntersection(Graph* graph, const Line2D& newEdgeLine, QueryResults* queryResults, int sourceIndex, int& edgeIndex, vml_vec2& closestIntersection, IntersectionType& intersectionType)
{
	float closestIntersectionDistance = FLT_MAX;
	edgeIndex = -1;
	intersectionType = NONE;

	for (unsigned int i = 0; i < queryResults->numResults; i++)
	{
		int queryEdgeIndex = queryResults->results[i];
		Edge& edge = graph->edges[queryEdgeIndex];

		// avoid intersecting parent or sibling
		if (edge.destination == sourceIndex || edge.source == sourceIndex)
		{
			continue;
		}

		vml_vec2 sourceVertexPosition = getPosition(graph, edge.source);
		vml_vec2 destinationVertexPosition = getPosition(graph, edge.destination);
		
		vml_vec2 intersection;
		Line2D edgeLine(sourceVertexPosition, destinationVertexPosition);

		if (newEdgeLine.intersects2(edgeLine, intersection))
		{
			float distance = vml_distance(newEdgeLine.getStart(), intersection);

			if (distance < closestIntersectionDistance)
			{
				if (vml_distance(sourceVertexPosition, intersection) <= graph->snapRadius)
				{
					intersectionType = SOURCE;
				}

				else if (vml_distance(destinationVertexPosition, intersection) <= graph->snapRadius)
				{
					intersectionType = DESTINATION;
				}

				else
				{
					intersectionType = EDGE;
				}

				closestIntersectionDistance = distance;
				closestIntersection = intersection;
				edgeIndex = queryEdgeIndex;
			}
		}

#ifdef _DEBUG
		// FIXME: use 64bits atomic operations!
		ATOMIC_ADD(graph->numCollisionChecks, unsigned int, 1);
#endif
	}

	return (intersectionType != NONE);
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE bool checkSnapping(Graph* graph, const Circle2D& snapCircle, QueryResults* queryResults, int sourceIndex, vml_vec2& closestSnapping, int& edgeIndex)
{
	edgeIndex = -1;
	float closestSnappingDistance = FLT_MAX;

	// check for snapping
	for (unsigned int i = 0; i < queryResults->numResults; i++)
	{
		int queryEdgeIndex = queryResults->results[i];
		Edge& edge = graph->edges[queryEdgeIndex];

		// avoid snapping parent or sibling
		if (edge.destination == sourceIndex || edge.source == sourceIndex)
		{
			continue;
		}

		vml_vec2 sourceVertexPosition = getPosition(graph, edge.source);
		vml_vec2 destinationVertexPosition = getPosition(graph, edge.destination);
		Line2D edgeLine(sourceVertexPosition, destinationVertexPosition);

		vml_vec2 intersection1;
		vml_vec2 intersection2;
		vml_vec2 snapping;
		int intersectionMask = edgeLine.intersects4(snapCircle, intersection1, intersection2);

		if (intersectionMask > 0)
		{
			float distance;

			if (intersectionMask == 1)
			{
				distance = vml_distance(snapCircle.getCenter(), intersection1);
				snapping = intersection1;
			}

			else if (intersectionMask == 2)
			{
				distance = vml_distance(snapCircle.getCenter(), intersection2);
				snapping = intersection2;
			}

			else if (intersectionMask == 3)
			{
				float distance1 = vml_distance(snapCircle.getCenter(), intersection1);
				float distance2 = vml_distance(snapCircle.getCenter(), intersection2);

				if (distance1 <= distance2)
				{
					snapping = intersection1;
					distance = distance1;
				}

				else
				{
					snapping = intersection2;
					distance = distance2;
				}
			}

			else
			{
				// FIXME: checking invariants
				THROW_EXCEPTION("invalid intersection mask");
			}

			if (distance < closestSnappingDistance)
			{
				closestSnappingDistance = distance;
				closestSnapping = snapping;
				edgeIndex = queryEdgeIndex;
			}
		}

#ifdef _DEBUG
		// FIXME: use 64bits atomic operations!
		ATOMIC_ADD(graph->numCollisionChecks, unsigned int, 1);
#endif
	}

	return (edgeIndex != -1);
}
#else
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE bool addRoad(Graph* graph, int sourceIndex, const vml_vec2& direction, int& newVertexIndex, vml_vec2& end, bool highway)
{
	vml_vec2 start = getPosition(graph, sourceIndex);
	end = start + direction;
	int edgeIndex;
	vml_vec2 snapping;
	vml_vec2 intersection;
	IntersectionType intersectionType;
	Line2D newEdgeLine(start, end);

	// critical section
	bool returnValue = false;
	bool leaveLoop = false;
	while (!leaveLoop)
	{ 
		if (ATOMIC_EXCH(graph->owner, int, THREAD_IDX_X) == -1)
		{
			if (checkIntersection(graph, newEdgeLine, sourceIndex, edgeIndex, intersection, intersectionType))
			{
				end = intersection;

				Edge& edge = graph->edges[edgeIndex];
				int edgeSource = edge.source;
				int edgeDestination = edge.destination;

				if (intersectionType == SOURCE)
				{
					newVertexIndex = edge.source;
					connect(graph, sourceIndex, newVertexIndex, highway);
				}

				else if (intersectionType == DESTINATION)
				{
					newVertexIndex = edge.destination;
					connect(graph, sourceIndex, newVertexIndex, highway);
				}

				else if (intersectionType == EDGE)
				{
					newVertexIndex = createVertex(graph, end);
					splitEdge(graph, edgeIndex, newVertexIndex);
					if (!connect(graph, sourceIndex, newVertexIndex, highway))
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

				returnValue = true;
			}

			else
			{
				Circle2D snapCircle(end, graph->snapRadius);

				if (checkSnapping(graph, snapCircle, sourceIndex, snapping, edgeIndex))
				{
					end = snapping;
					newVertexIndex = createVertex(graph, end);
					splitEdge(graph, edgeIndex, newVertexIndex);
					if (!connect(graph, sourceIndex, newVertexIndex, highway))
					{
						// FIXME: checking invariants
						THROW_EXCEPTION("unexpected situation");
					}

					returnValue = true;
				}

				else
				{
					newVertexIndex = createVertex(graph, end);
					if (connect(graph, sourceIndex, newVertexIndex, highway))
					{
						returnValue = false;
					}
					else
					{
						returnValue = true;
					}
				}
			}

			THREADFENCE();

			leaveLoop = true;
			ATOMIC_EXCH(graph->owner, int, -1);
		}
	}

	return returnValue;
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE bool checkIntersection(Graph* graph, const Line2D& newEdgeLine, int sourceIndex, int& edgeIndex, vml_vec2& closestIntersection, IntersectionType& intersectionType)
{
	// check for intersections
	float closestIntersectionDistance = FLT_MAX;
	edgeIndex = -1;
	intersectionType = NONE;

	for (int i = 0; i < graph->numEdges; i++)
	{
		Edge& edge = graph->edges[i];

		// avoid intersecting parent or sibling
		if (edge.destination == sourceIndex || edge.source == sourceIndex)
		{
			continue;
		}

		vml_vec2 sourceVertexPosition = getPosition(graph, edge.source);
		vml_vec2 destinationVertexPosition = getPosition(graph, edge.destination);
		vml_vec2 intersection;
		Line2D edgeLine(sourceVertexPosition, destinationVertexPosition);

		if (newEdgeLine.intersects2(edgeLine, intersection))
		{
			float distance = vml_distance(newEdgeLine.getStart(), intersection);

			if (distance < closestIntersectionDistance)
			{
				if (vml_distance(sourceVertexPosition, intersection) <= graph->snapRadius)
				{
					intersectionType = SOURCE;
				}

				else if (vml_distance(destinationVertexPosition, intersection) <= graph->snapRadius)
				{
					intersectionType = DESTINATION;
				}

				else
				{
					intersectionType = EDGE;
				}

				closestIntersectionDistance = distance;
				closestIntersection = intersection;
				edgeIndex = i;
			}
		}

#ifdef _DEBUG
		ATOMIC_ADD(graph->numCollisionChecks, unsigned int, 1);
#endif
	}

	return (intersectionType != NONE);
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE bool checkSnapping(Graph* graph, const Circle2D& snapCircle, int sourceIndex, vml_vec2& closestSnapping, int& edgeIndex)
{
	edgeIndex = -1;
	float closestSnappingDistance = FLT_MAX;

	// check for snapping
	for (int i = 0; i < graph->numEdges; i++)
	{
		Edge& edge = graph->edges[i];

		// avoid snapping parent or sibling
		if (edge.destination == sourceIndex || edge.source == sourceIndex)
		{
			continue;
		}

		vml_vec2 sourceVertexPosition = getPosition(graph, edge.source);
		vml_vec2 destinationVertexPosition = getPosition(graph, edge.destination);
		Line2D edgeLine(sourceVertexPosition, destinationVertexPosition);
		vml_vec2 intersection1;
		vml_vec2 intersection2;
		vml_vec2 snapping;
		int intersectionMask = edgeLine.intersects4(snapCircle, intersection1, intersection2);

		if (intersectionMask > 0)
		{
			float distance;

			if (intersectionMask == 1)
			{
				distance = vml_distance(snapCircle.getCenter(), intersection1);
				snapping = intersection1;
			}

			else if (intersectionMask == 2)
			{
				distance = vml_distance(snapCircle.getCenter(), intersection2);
				snapping = intersection2;
			}

			else if (intersectionMask == 3)
			{
				float distance1 = vml_distance(snapCircle.getCenter(), intersection1);
				float distance2 = vml_distance(snapCircle.getCenter(), intersection2);

				if (distance1 <= distance2)
				{
					snapping = intersection1;
					distance = distance1;
				}

				else
				{
					snapping = intersection2;
					distance = distance2;
				}
			}

			else
			{
				// FIXME: checking invariants
				THROW_EXCEPTION("invalid intersection mask");
			}

			if (distance < closestSnappingDistance)
			{
				closestSnappingDistance = distance;
				closestSnapping = snapping;
				edgeIndex = i;
			}
		}

#ifdef _DEBUG
		ATOMIC_ADD(graph->numCollisionChecks, unsigned int, 1);
#endif
	}

	return (edgeIndex != -1);
}
#endif

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