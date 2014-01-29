#ifndef ROADNETWORKGRAPH_GRAPH_CUH
#define ROADNETWORKGRAPH_GRAPH_CUH

#include "Defines.h"
#include <BaseGraph.cuh>
#include <QuadTree.cuh>

#include <Box2D.cuh>
#include <Line2D.cuh>
#include <Circle2D.cuh>

#include <vector_math.h>

namespace RoadNetworkGraph
{

//////////////////////////////////////////////////////////////////////////
struct GraphTraversal;

//////////////////////////////////////////////////////////////////////////
enum IntersectionType
{
	NONE,
	SOURCE,
	DESTINATION,
	EDGE
};

//////////////////////////////////////////////////////////////////////////
struct Graph : public BaseGraph
{
	unsigned int maxVertices;
	unsigned int maxEdges;
	unsigned int maxResultsPerQuery;
	float snapRadius;
#ifdef _DEBUG
	unsigned long numCollisionChecks;
#endif
#ifdef USE_QUADTREE
	QuadTree* quadtree;
	EdgeIndex* queryResult;
#endif

};

/*
//////////////////////////////////////////////////////////////////////////
#ifdef USE_QUADTREE
HOST_CODE void initializeGraphOnHost(Graph* graph, float snapRadius, unsigned int maxVertices, unsigned int maxEdges, Vertex* vertices, Edge* edges, QuadTree* quadtree, unsigned int maxResultsPerQuery, EdgeIndex* queryResult);
GLOBAL_CODE void initializeGraphOnDevice(Graph* graph, float snapRadius, unsigned int maxVertices, unsigned int maxEdges, Vertex* vertices, Edge* edges, QuadTree* quadtree, unsigned int maxResultsPerQuery, EdgeIndex* queryResult);
#else
HOST_CODE void initializeGraphOnHost(Graph* graph, float snapRadius, unsigned int maxVertices, unsigned int maxEdges, Vertex* vertices, Edge* edges);
GLOBAL_CODE void initializeGraphOnDevice(Graph* graph, float snapRadius, unsigned int maxVertices, unsigned int maxEdges, Vertex* vertices, Edge* edges);
#endif
//////////////////////////////////////////////////////////////////////////
GLOBAL_CODE void updateNonPointerFields(Graph* graph, unsigned int maxVertices, unsigned int maxEdges, unsigned int maxResultsPerQuery, unsigned int numVertices, unsigned int numEdges
#ifdef _DEBUG
	, unsigned long numCollisionChecks
#endif	
);
//////////////////////////////////////////////////////////////////////////
HOST_CODE void copy(Graph* graph, BaseGraph* other);
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE vml_vec2 getPosition(Graph* graph, VertexIndex vertexIndex);
//////////////////////////////////////////////////////////////////////////
HOST_AND_DEVICE_CODE VertexIndex createVertex(Graph* graph, const vml_vec2& position);
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE bool addRoad(Graph* graph, VertexIndex sourceIndex, const vml_vec2& direction, VertexIndex& newVertexIndex, vml_vec2& end, bool highway);
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE bool connect(Graph* graph, VertexIndex source, VertexIndex destination, bool highway);
//////////////////////////////////////////////////////////////////////////
HOST_CODE void removeDeadEndRoads(Graph* graph);
//////////////////////////////////////////////////////////////////////////
HOST_CODE void traverse(const Graph* graph, GraphTraversal& traversal);

#ifdef _DEBUG
//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getAllocatedVertices(Graph* graph);
//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getVerticesInUse(Graph* graph);
//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getAllocatedEdges(Graph* graph);
//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getEdgesInUse(Graph* graph);
//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getMaxVertexInConnections(Graph* graph);
//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getMaxVertexInConnectionsInUse(Graph* graph);
//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getMaxVertexOutConnections(Graph* graph);
//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getMaxVertexOutConnectionsInUse(Graph* graph);
//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getAverageVertexInConnectionsInUse(Graph* graph);
//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getAverageVertexOutConnectionsInUse(Graph* graph);
//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getAllocatedMemory(Graph* graph);
//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getMemoryInUse(Graph* graph);
//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned long getNumCollisionChecks(Graph* graph);
#endif

*/

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void splitEdge(Graph* graph, EdgeIndex edge, VertexIndex vertex);
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE bool checkIntersection(Graph* graph, const Line2D& newEdgeLine, unsigned int querySize, VertexIndex source, EdgeIndex& edgeIndex, vml_vec2& closestIntersection, IntersectionType& intersectionType);
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE bool checkSnapping(Graph* graph, const Circle2D& snapCircle, unsigned int querySize, VertexIndex source, vml_vec2& closestSnapping, EdgeIndex& edgeIndex);

#ifdef USE_QUADTREE
//////////////////////////////////////////////////////////////////////////
GLOBAL_CODE void initializeGraphOnDevice(Graph* graph, float snapRadius, unsigned int maxVertices, unsigned int maxEdges, Vertex* vertices, Edge* edges, QuadTree* quadtree, unsigned int maxResultsPerQuery, EdgeIndex* queryResult)
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
HOST_CODE void initializeGraphOnHost(Graph* graph, float snapRadius, unsigned int maxVertices, unsigned int maxEdges, Vertex* vertices, Edge* edges, QuadTree* quadtree, unsigned int maxResultsPerQuery, EdgeIndex* queryResult);
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
DEVICE_CODE vml_vec2 getPosition(Graph* graph, VertexIndex vertexIndex)
{
	// FIXME: checking invariants
	if (vertexIndex >= graph->numVertices)
	{
		THROW_EXCEPTION("invalid vertexIndex");
	}

	return graph->vertices[vertexIndex].getPosition();
}

//////////////////////////////////////////////////////////////////////////
HOST_AND_DEVICE_CODE VertexIndex createVertex(Graph* graph, const vml_vec2& position)
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
DEVICE_CODE bool connect(Graph* graph, VertexIndex sourceVertexIndex, VertexIndex destinationVertexIndex, bool highway)
{
	Vertex& sourceVertex = graph->vertices[sourceVertexIndex];
	Vertex& destinationVertex = graph->vertices[destinationVertexIndex];

	// TODO: unidirectional graph -> get rid of Ins and Outs avoiding duplicate edges
	for (unsigned int i = 0; i < sourceVertex.numAdjacencies; i++)
	{
		if (sourceVertex.adjacencies[i] == destinationVertexIndex)
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
	newEdge.source = sourceVertexIndex;
	newEdge.destination = destinationVertexIndex;
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

	sourceVertex.adjacencies[sourceVertex.numAdjacencies++] = destinationVertexIndex;

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

	destinationVertex.adjacencies[destinationVertex.numAdjacencies++] = sourceVertexIndex;

#ifdef USE_QUADTREE
	insert(graph->quadtree, graph->numEdges, Line2D(sourceVertex.getPosition(), destinationVertex.getPosition()));
#endif

	graph->numEdges++;

	return true;
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void splitEdge(Graph* graph, EdgeIndex edgeIndex, VertexIndex splitVertexIndex)
{
	Edge& edge = graph->edges[edgeIndex];
	Vertex& sourceVertex = graph->vertices[edge.source];
	VertexIndex oldDestinationVertexIndex = edge.destination;
	Vertex& oldDestinationVertex = graph->vertices[oldDestinationVertexIndex];
	
	edge.destination = splitVertexIndex;
	Vertex& splitVertex = graph->vertices[splitVertexIndex];

#ifdef USE_QUADTREE
	remove(graph->quadtree, edgeIndex, Line2D(sourceVertex.getPosition(), oldDestinationVertex.getPosition()));
	insert(graph->quadtree, edgeIndex, Line2D(sourceVertex.getPosition(), splitVertex.getPosition()));
#endif

	replaceAdjacency(sourceVertex, oldDestinationVertexIndex, splitVertexIndex);

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
	newEdge.source = splitVertexIndex;
	newEdge.destination = oldDestinationVertexIndex;
	newEdge.attr1 = edge.attr1;

	replaceAdjacency(oldDestinationVertex, edge.source, splitVertexIndex);
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

	splitVertex.adjacencies[splitVertex.numAdjacencies++] = oldDestinationVertexIndex;
	
#ifdef USE_QUADTREE
	insert(graph->quadtree, graph->numEdges, Line2D(splitVertex.getPosition(), oldDestinationVertex.getPosition()));
#endif
	graph->numEdges++;
}

#ifdef USE_QUADTREE
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE bool addRoad(Graph* graph, VertexIndex sourceIndex, const vml_vec2& direction, VertexIndex& newVertexIndex, vml_vec2& end, bool highway)
{
	vml_vec2 start = getPosition(graph, sourceIndex);
	end = start + direction;
	Line2D newEdgeLine(start, end);
	Box2D newEdgeBounds(newEdgeLine);
	unsigned int querySize;
	query(graph->quadtree, newEdgeBounds, graph->queryResult, querySize);
	EdgeIndex edgeIndex;
	vml_vec2 snapping;
	vml_vec2 intersection;
	IntersectionType intersectionType;

	if (checkIntersection(graph, newEdgeLine, querySize, sourceIndex, edgeIndex, intersection, intersectionType))
	{
		end = intersection;

		if (intersectionType == SOURCE)
		{
			newVertexIndex = graph->edges[edgeIndex].source;
			connect(graph, sourceIndex, newVertexIndex, highway);
		}

		else if (intersectionType == DESTINATION)
		{
			newVertexIndex = graph->edges[edgeIndex].destination;
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

		return true;
	}

	else
	{
		Circle2D snapCircle(end, graph->snapRadius);
		query(graph->quadtree, snapCircle, graph->queryResult, querySize);

		if (checkSnapping(graph, snapCircle, querySize, sourceIndex, snapping, edgeIndex))
		{
			end = snapping;
			newVertexIndex = createVertex(graph, end);
			splitEdge(graph, edgeIndex, newVertexIndex);
			if (!connect(graph, sourceIndex, newVertexIndex, highway))
			{
				// FIXME: checking invariants
				THROW_EXCEPTION("unexpected situation");
			}
			return true;
		}

		else
		{
			newVertexIndex = createVertex(graph, end);
			if (connect(graph, sourceIndex, newVertexIndex, highway))
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
DEVICE_CODE bool checkIntersection(Graph* graph, const Line2D& newEdgeLine, unsigned int querySize, VertexIndex sourceIndex, EdgeIndex& edgeIndex, vml_vec2& closestIntersection, IntersectionType& intersectionType)
{
	float closestIntersectionDistance = FLT_MAX;
	edgeIndex = -1;
	intersectionType = NONE;

	for (unsigned int i = 0; i < querySize; i++)
	{
		EdgeIndex queryEdgeIndex = graph->queryResult[i];
		Edge& edge = graph->edges[queryEdgeIndex];

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
				edgeIndex = queryEdgeIndex;
			}
		}

#ifdef _DEBUG
		graph->numCollisionChecks++;
#endif
	}

	return (intersectionType != NONE);
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE bool checkSnapping(Graph* graph, const Circle2D& snapCircle, unsigned int querySize, VertexIndex sourceIndex, vml_vec2& closestSnapping, EdgeIndex& edgeIndex)
{
	edgeIndex = -1;
	float closestSnappingDistance = FLT_MAX;

	// check for snapping
	for (unsigned int i = 0; i < querySize; i++)
	{
		EdgeIndex queryEdgeIndex = graph->queryResult[i];
		Edge& edge = graph->edges[queryEdgeIndex];

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
				edgeIndex = queryEdgeIndex;
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
DEVICE_CODE bool addRoad(Graph* graph, VertexIndex sourceIndex, const vml_vec2& direction, VertexIndex& newVertexIndex, vml_vec2& end, bool highway)
{
	vml_vec2 start = getPosition(graph, sourceIndex);
	end = start + direction;
	EdgeIndex edgeIndex;
	vml_vec2 snapping;
	vml_vec2 intersection;
	IntersectionType intersectionType;

	if (checkIntersection(graph, start, end, sourceIndex, edgeIndex, intersection, intersectionType))
	{
		end = intersection;

		if (intersectionType == SOURCE)
		{
			newVertexIndex = graph->edges[edgeIndex].source;
		}

		else if (intersectionType == DESTINATION)
		{
			newVertexIndex = graph->edges[edgeIndex].destination;
		}

		else if (intersectionType == EDGE)
		{
			newVertexIndex = createVertex(graph, end);
			splitEdge(graph, edgeIndex, newVertexIndex);
		}

		else
		{
			// FIXME: checking invariants
			THROW_EXCEPTION("unknown intersection type");
		}

		return connect(graph, sourceIndex, newVertexIndex, highway);
	}

	else
	{
		Circle2D snapCircle(end, graph->snapRadius);

		if (checkSnapping(graph, snapCircle, sourceIndex, snapping, edgeIndex))
		{
			end = snapping;
			newVertexIndex = createVertex(graph, end);
			splitEdge(graph, edgeIndex, newVertexIndex);
			return connect(graph, sourceIndex, newVertexIndex, highway);
		}

		else
		{
			newVertexIndex = createVertex(graph, end);
			return connect(graph, sourceIndex, newVertexIndex, highway);
		}
	}
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE bool checkIntersection(Graph* graph, const vml_vec2& start, const vml_vec2& end, VertexIndex sourceIndex, EdgeIndex& edgeIndex, vml_vec2& closestIntersection, IntersectionType& intersectionType)
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
DEVICE_CODE bool checkSnapping(Graph* graph, const Circle2D& snapCircle, VertexIndex sourceIndex, vml_vec2& closestSnapping, EdgeIndex& edgeIndex)
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

//////////////////////////////////////////////////////////////////////////
HOST_CODE void removeDeadEndRoads(Graph* graph);
//////////////////////////////////////////////////////////////////////////
HOST_CODE void traverse(const Graph* graph, GraphTraversal& traversal);

#ifdef _DEBUG
//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getAllocatedVertices(Graph* graph);
//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getVerticesInUse(Graph* graph);
//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getAllocatedEdges(Graph* graph);
//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getEdgesInUse(Graph* graph);
//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getMaxVertexInConnections(Graph* graph);
//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getMaxVertexInConnectionsInUse(Graph* graph);
//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getMaxVertexOutConnections(Graph* graph);
//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getMaxVertexOutConnectionsInUse(Graph* graph);
//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getAverageVertexInConnectionsInUse(Graph* graph);
//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getAverageVertexOutConnectionsInUse(Graph* graph);
//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getAllocatedMemory(Graph* graph);
//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getMemoryInUse(Graph* graph);
//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned long getNumCollisionChecks(Graph* graph);
#endif

}

#endif