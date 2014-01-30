#ifndef GRAPHFUNCTIONS_CUH
#define GRAPHFUNCTIONS_CUH

#pragma once

#include <Constants.h>
#include <CpuGpuCompatibility.h>
#include <Graph.h>
#include <GraphTraversal.h>
#include <QuadTree.h>
#include <Box2D.h>
#include <Line2D.h>
#include <Circle2D.h>
#include <VectorMath.h>
#include <QuadTreeFunctions.cuh>
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
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE bool checkIntersection(Graph* graph, const Line2D& newEdgeLine, unsigned int querySize, int source, int& edgeIndex, vml_vec2& closestIntersection, IntersectionType& intersectionType);
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE bool checkSnapping(Graph* graph, const Circle2D& snapCircle, unsigned int querySize, int source, vml_vec2& closestSnapping, int& edgeIndex);

#ifdef USE_QUADTREE
//////////////////////////////////////////////////////////////////////////
GLOBAL_CODE void initializeGraphOnDevice(Graph* graph, float snapRadius, unsigned int maxVertices, unsigned int maxEdges, Vertex* vertices, Edge* edges, QuadTree* quadtree, unsigned int maxResultsPerQuery, int* queryResult)
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

//////////////////////////////////////////////////////////////////////////
HOST_CODE void initializeGraphOnHost(Graph* graph, float snapRadius, unsigned int maxVertices, unsigned int maxEdges, Vertex* vertices, Edge* edges, QuadTree* quadtree, unsigned int maxResultsPerQuery, int* queryResult);
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
#ifdef _DEBUG
	graph->numCollisionChecks = 0;
#endif
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE void initializeGraphOnHost(Graph* graph, float snapRadius, unsigned int maxVertices, unsigned int maxEdges, Vertex* vertices, Edge* edges);
#endif

//////////////////////////////////////////////////////////////////////////
GLOBAL_CODE void updateNonPointerFields(Graph* graph, unsigned int numVertices, unsigned int numEdges, unsigned int maxVertices, unsigned int maxEdges, unsigned int maxResultsPerQuery
#ifdef _DEBUG
	, unsigned long numCollisionChecks
#endif	
)
{
	graph->numVertices = numVertices;
	graph->numEdges = numEdges;
	graph->maxVertices = maxVertices;
	graph->maxEdges = maxEdges;
	graph->maxResultsPerQuery = maxResultsPerQuery;
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
		THROW_EXCEPTION("invalid vertexIndex");
	}

	return graph->vertices[vertexIndex].getPosition();
}

//////////////////////////////////////////////////////////////////////////
HOST_AND_DEVICE_CODE int createVertex(Graph* graph, const vml_vec2& position)
{
	// FIXME: checking boundaries
	if (graph->numVertices >= (int)graph->maxVertices)
	{
		THROW_EXCEPTION("max. vertices overflow");
	}

	Vertex& newVertex = graph->vertices[graph->numVertices];
	newVertex.index = graph->numVertices;
	newVertex.setPosition(position);
	return graph->numVertices++;
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE bool connect(Graph* graph, int sourceint, int destinationint, bool highway)
{
	Vertex& sourceVertex = graph->vertices[sourceint];
	Vertex& destinationVertex = graph->vertices[destinationint];

	// TODO: unidirectional graph -> get rid of Ins and Outs avoiding duplicate edges
	for (unsigned int i = 0; i < sourceVertex.numAdjacencies; i++)
	{
		if (sourceVertex.adjacencies[i] == destinationint)
		{
			return false;
		}
	}

	// FIXME: checking boundaries
	if (graph->numEdges >= (int)graph->maxEdges)
	{
		THROW_EXCEPTION("max. edges overflow");
	}

	Edge& newEdge = graph->edges[graph->numEdges];
	newEdge.index = graph->numEdges;
	newEdge.source = sourceint;
	newEdge.destination = destinationint;
	newEdge.attr1 = (highway) ? 1 : 0;

	// FIXME: checking boundaries
	if (sourceVertex.numOuts >= MAX_VERTEX_OUT_CONNECTIONS)
	{
		THROW_EXCEPTION("max. vertex connections (out) overflow");
	}

	sourceVertex.outs[sourceVertex.numOuts++] = graph->numEdges;

	// FIXME: checking boundaries
	if (sourceVertex.numAdjacencies >= MAX_VERTEX_ADJACENCIES)
	{
		THROW_EXCEPTION("max. vertex adjacencies overflow");
	}

	sourceVertex.adjacencies[sourceVertex.numAdjacencies++] = destinationint;

	// FIXME: checking boundaries
	if (destinationVertex.numIns >= MAX_VERTEX_IN_CONNECTIONS)
	{
		THROW_EXCEPTION("max. vertex connections (in) overflow");
	}

	destinationVertex.ins[destinationVertex.numIns++] = graph->numEdges;

	// FIXME: checking boundaries
	if (destinationVertex.numAdjacencies >= MAX_VERTEX_ADJACENCIES)
	{
		THROW_EXCEPTION("max. vertex adjacencies overflow");
	}

	destinationVertex.adjacencies[destinationVertex.numAdjacencies++] = sourceint;

#ifdef USE_QUADTREE
	insert(graph->quadtree, graph->numEdges, Line2D(sourceVertex.getPosition(), destinationVertex.getPosition()));
#endif

	graph->numEdges++;

	return true;
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void splitEdge(Graph* graph, int edgeIndex, int splitint)
{
	Edge& edge = graph->edges[edgeIndex];
	Vertex& sourceVertex = graph->vertices[edge.source];
	int oldDestinationint = edge.destination;
	Vertex& oldDestinationVertex = graph->vertices[oldDestinationint];
	
	edge.destination = splitint;
	Vertex& splitVertex = graph->vertices[splitint];

#ifdef USE_QUADTREE
	remove(graph->quadtree, edgeIndex, Line2D(sourceVertex.getPosition(), oldDestinationVertex.getPosition()));
	insert(graph->quadtree, edgeIndex, Line2D(sourceVertex.getPosition(), splitVertex.getPosition()));
#endif

	replaceAdjacency(sourceVertex, oldDestinationint, splitint);

	// FIXME: checking boundaries
	if (splitVertex.numIns >= MAX_VERTEX_IN_CONNECTIONS)
	{
		THROW_EXCEPTION("max. vertex connections (in) overflow");
	}

	splitVertex.ins[splitVertex.numIns++] = edgeIndex;

	// FIXME: checking boundaries
	if (splitVertex.numAdjacencies >= MAX_VERTEX_ADJACENCIES)
	{
		THROW_EXCEPTION("max. vertex adjacencies overflow");
	}

	splitVertex.adjacencies[splitVertex.numAdjacencies++] = edge.source;

	// FIXME: checking boundaries
	if (graph->numEdges >= (int)graph->maxEdges)
	{
		THROW_EXCEPTION("max. edges overflow");
	}

	Edge& newEdge = graph->edges[graph->numEdges];
	newEdge.index = graph->numEdges;
	newEdge.source = splitint;
	newEdge.destination = oldDestinationint;
	newEdge.attr1 = edge.attr1;

	replaceAdjacency(oldDestinationVertex, edge.source, splitint);
	replaceInEdge(oldDestinationVertex, edgeIndex, graph->numEdges);

	// FIXME: checking boundaries
	if (splitVertex.numOuts >= MAX_VERTEX_OUT_CONNECTIONS)
	{
		THROW_EXCEPTION("max. vertex connections (out) overflow");
	}

	splitVertex.outs[splitVertex.numOuts++] = graph->numEdges;

	// FIXME: checking boundaries
	if (splitVertex.numAdjacencies >= MAX_VERTEX_ADJACENCIES)
	{
		THROW_EXCEPTION("max. vertex adjacencies overflow");
	}

	splitVertex.adjacencies[splitVertex.numAdjacencies++] = oldDestinationint;
	
#ifdef USE_QUADTREE
	insert(graph->quadtree, graph->numEdges, Line2D(splitVertex.getPosition(), oldDestinationVertex.getPosition()));
#endif
	graph->numEdges++;
}

#ifdef USE_QUADTREE
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE bool addRoad(Graph* graph, int sourceIndex, const vml_vec2& direction, int& newint, vml_vec2& end, bool highway)
{
	vml_vec2 start = getPosition(graph, sourceIndex);
	end = start + direction;
	Line2D newEdgeLine(start, end);
	Box2D newEdgeBounds(newEdgeLine);
	unsigned int querySize;
	query(graph->quadtree, newEdgeBounds, graph->queryResult, querySize);
	int edgeIndex;
	vml_vec2 snapping;
	vml_vec2 intersection;
	IntersectionType intersectionType;

	if (checkIntersection(graph, newEdgeLine, querySize, sourceIndex, edgeIndex, intersection, intersectionType))
	{
		end = intersection;

		if (intersectionType == SOURCE)
		{
			newint = graph->edges[edgeIndex].source;
			connect(graph, sourceIndex, newint, highway);
		}

		else if (intersectionType == DESTINATION)
		{
			newint = graph->edges[edgeIndex].destination;
			connect(graph, sourceIndex, newint, highway);
		}

		else if (intersectionType == EDGE)
		{
			newint = createVertex(graph, end);
			splitEdge(graph, edgeIndex, newint);
			if (!connect(graph, sourceIndex, newint, highway))
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

		return true;
	}

	else
	{
		Circle2D snapCircle(end, graph->snapRadius);
		query(graph->quadtree, snapCircle, graph->queryResult, querySize);

		if (checkSnapping(graph, snapCircle, querySize, sourceIndex, snapping, edgeIndex))
		{
			end = snapping;
			newint = createVertex(graph, end);
			splitEdge(graph, edgeIndex, newint);
			if (!connect(graph, sourceIndex, newint, highway))
			{
				// FIXME: checking invariants
				THROW_EXCEPTION("unexpected situation");
			}
			return true;
		}

		else
		{
			newint = createVertex(graph, end);
			if (connect(graph, sourceIndex, newint, highway))
			{
				return false;
			}
			else
			{
				return true;
			}
		}
	}
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE bool checkIntersection(Graph* graph, const Line2D& newEdgeLine, unsigned int querySize, int sourceIndex, int& edgeIndex, vml_vec2& closestIntersection, IntersectionType& intersectionType)
{
	float closestIntersectionDistance = FLT_MAX;
	edgeIndex = -1;
	intersectionType = NONE;

	for (unsigned int i = 0; i < querySize; i++)
	{
		int queryint = graph->queryResult[i];
		Edge& edge = graph->edges[queryint];

		// avoid intersecting parent or sibling
		if (edge.destination == sourceIndex || edge.source == sourceIndex)
		{
			continue;
		}

		Vertex& sourceVertex = graph->vertices[edge.source];
		Vertex& destinationVertex = graph->vertices[edge.destination];
		vml_vec2 intersection;
		Line2D edgeLine(sourceVertex.getPosition(), destinationVertex.getPosition());

		if (newEdgeLine.intersects(edgeLine, intersection))
		{
			float distance = vml_distance(newEdgeLine.getStart(), intersection);

			if (distance < closestIntersectionDistance)
			{
				if (vml_distance(sourceVertex.getPosition(), intersection) <= graph->snapRadius)
				{
					intersectionType = SOURCE;
				}

				else if (vml_distance(destinationVertex.getPosition(), intersection) <= graph->snapRadius)
				{
					intersectionType = DESTINATION;
				}

				else
				{
					intersectionType = EDGE;
				}

				closestIntersectionDistance = distance;
				closestIntersection = intersection;
				edgeIndex = queryint;
			}
		}

#ifdef _DEBUG
		graph->numCollisionChecks++;
#endif
	}

	return (intersectionType != NONE);
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE bool checkSnapping(Graph* graph, const Circle2D& snapCircle, unsigned int querySize, int sourceIndex, vml_vec2& closestSnapping, int& edgeIndex)
{
	edgeIndex = -1;
	float closestSnappingDistance = FLT_MAX;

	// check for snapping
	for (unsigned int i = 0; i < querySize; i++)
	{
		int queryint = graph->queryResult[i];
		Edge& edge = graph->edges[queryint];

		// avoid snapping parent or sibling
		if (edge.destination == sourceIndex || edge.source == sourceIndex)
		{
			continue;
		}

		Vertex& sourceVertex = graph->vertices[edge.source];
		Vertex& destinationVertex = graph->vertices[edge.destination];
		Line2D edgeLine(sourceVertex.getPosition(), destinationVertex.getPosition());
		vml_vec2 intersection1;
		vml_vec2 intersection2;
		vml_vec2 snapping;
		int intersectionMask = edgeLine.intersects(snapCircle, intersection1, intersection2);

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
				edgeIndex = queryint;
			}
		}

#ifdef _DEBUG
		graph->numCollisionChecks++;
#endif
	}

	return (edgeIndex != -1);
}
#else
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE bool addRoad(Graph* graph, int sourceIndex, const vml_vec2& direction, int& newint, vml_vec2& end, bool highway)
{
	vml_vec2 start = getPosition(graph, sourceIndex);
	end = start + direction;
	int edgeIndex;
	vml_vec2 snapping;
	vml_vec2 intersection;
	IntersectionType intersectionType;

	if (checkIntersection(graph, start, end, sourceIndex, edgeIndex, intersection, intersectionType))
	{
		end = intersection;

		if (intersectionType == SOURCE)
		{
			newint = graph->edges[edgeIndex].source;
		}

		else if (intersectionType == DESTINATION)
		{
			newint = graph->edges[edgeIndex].destination;
		}

		else if (intersectionType == EDGE)
		{
			newint = createVertex(graph, end);
			splitEdge(graph, edgeIndex, newint);
		}

		else
		{
			// FIXME: checking invariants
			THROW_EXCEPTION("unknown intersection type");
		}

		return connect(graph, sourceIndex, newint, highway);
	}

	else
	{
		Circle2D snapCircle(end, graph->snapRadius);

		if (checkSnapping(graph, snapCircle, sourceIndex, snapping, edgeIndex))
		{
			end = snapping;
			newint = createVertex(graph, end);
			splitEdge(graph, edgeIndex, newint);
			return connect(graph, sourceIndex, newint, highway);
		}

		else
		{
			newint = createVertex(graph, end);
			return connect(graph, sourceIndex, newint, highway);
		}
	}
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE bool checkIntersection(Graph* graph, const vml_vec2& start, const vml_vec2& end, int sourceIndex, int& edgeIndex, vml_vec2& closestIntersection, IntersectionType& intersectionType)
{
	// check for intersections
	float closestIntersectionDistance = FLT_MAX;
	edgeIndex = -1;
	intersectionType = NONE;
	Line2D newEdgeLine(start, end);

	for (int i = 0; i < graph->numEdges; i++)
	{
		Edge& edge = graph->edges[i];

		// avoid intersecting parent or sibling
		if (edge.destination == sourceIndex || edge.source == sourceIndex)
		{
			continue;
		}

		Vertex& sourceVertex = graph->vertices[edge.source];
		Vertex& destinationVertex = graph->vertices[edge.destination];
		vml_vec2 intersection;
		Line2D edgeLine(sourceVertex.getPosition(), destinationVertex.getPosition());

		if (newEdgeLine.intersects(edgeLine, intersection))
		{
			float distance = vml_distance(start, intersection);

			if (distance < closestIntersectionDistance)
			{
				if (vml_distance(sourceVertex.getPosition(), intersection) <= graph->snapRadius)
				{
					intersectionType = SOURCE;
				}

				else if (vml_distance(destinationVertex.getPosition(), intersection) <= graph->snapRadius)
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
		graph->numCollisionChecks++;
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

		Vertex& sourceVertex = graph->vertices[edge.source];
		Vertex& destinationVertex = graph->vertices[edge.destination];
		Line2D edgeLine(sourceVertex.getPosition(), destinationVertex.getPosition());
		vml_vec2 intersection1;
		vml_vec2 intersection2;
		vml_vec2 snapping;
		int intersectionMask = edgeLine.intersects(snapCircle, intersection1, intersection2);

		if (intersectionMask > 0)
		{
			float distance;

			if (intersectionMask == 1)
			{
				distance = vml_distance(snapCircle.center, intersection1);
				snapping = intersection1;
			}

			else if (intersectionMask == 2)
			{
				distance = vml_distance(snapCircle.center, intersection2);
				snapping = intersection2;
			}

			else if (intersectionMask == 3)
			{
				float distance1 = vml_distance(snapCircle.center, intersection1);
				float distance2 = vml_distance(snapCircle.center, intersection2);

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
		graph->numCollisionChecks++;
#endif
	}

	return (edgeIndex != -1);
}
#endif

#ifdef _DEBUG
#ifdef USE_QUADTREE
//////////////////////////////////////////////////////////////////////////
HOST_CODE void initializeGraphOnHost(Graph* graph, float snapRadius, unsigned int maxVertices, unsigned int maxEdges, Vertex* vertices, Edge* edges, QuadTree* quadtree, unsigned int maxResultsPerQuery, int* queryResult)
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

#endif