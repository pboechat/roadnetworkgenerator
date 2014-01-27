#include <Graph.h>
#include <QuadTree.h>
#include <GraphTraversal.h>

#include <cfloat>
#include <exception>

namespace RoadNetworkGraph
{

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void splitEdge(Graph* graph, EdgeIndex edge, VertexIndex vertex);
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE bool checkIntersection(Graph* graph, const Line2D& newEdgeLine, unsigned int querySize, VertexIndex source, EdgeIndex& edgeIndex, vml_vec2& closestIntersection, IntersectionType& intersectionType);
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE bool checkSnapping(Graph* graph, const Circle2D& snapCircle, unsigned int querySize, VertexIndex source, vml_vec2& closestSnapping, EdgeIndex& edgeIndex);
//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getValency(Graph* graph, const Vertex& vertex);

#ifdef USE_QUADTREE
//////////////////////////////////////////////////////////////////////////
GLOBAL_CODE void initializeGraph(Graph* graph, float snapRadius, unsigned int maxVertices, unsigned int maxEdges, Vertex* vertices, Edge* edges, QuadTree* quadtree, unsigned int maxResultsPerQuery, EdgeIndex* queryResult)
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
GLOBAL_CODE void initializeGraph(Graph* graph, float snapRadius, unsigned int maxVertices, unsigned int maxEdges, Vertex* vertices, Edge* edges)
{
	graph->numVertices = 0;
	graph->numEdges = 0;
	graph->vertices = vertices;
	graph->edges = edges;
	graph->maxVertices = maxVertices;
	graph->maxEdges = maxEdges;
	graph->maxResultsPerQuery = maxResultsPerQuery;
	graph->snapRadius = snapRadius;
#ifdef _DEBUG
	graph->numCollisionChecks = 0;
#endif
}
#endif

//////////////////////////////////////////////////////////////////////////
GLOBAL_CODE void updateNumVerticesAndNumEdges(Graph* graph, unsigned int numVertices, unsigned int numEdges)
{
	graph->numVertices = numVertices;
	graph->numEdges = numEdges;
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE void copy(Graph* graph, BaseGraph* other)
{
	other->numEdges = graph->numEdges;
	other->numVertices = graph->numVertices;
}

//////////////////////////////////////////////////////////////////////////
HOST_AND_DEVICE_CODE vml_vec2 getPosition(Graph* graph, VertexIndex vertexIndex)
{
	// FIXME: checking invariants
	if (vertexIndex >= graph->numVertices)
	{
		THROW_EXCEPTION("invalid vertexIndex");
	}

	return graph->vertices[vertexIndex].position;
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
	newVertex.position = position;
	return graph->numVertices++;
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
	insert(graph->quadtree, graph->numEdges, Line2D(sourceVertex.position, destinationVertex.position));
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
	remove(graph->quadtree, edgeIndex, Line2D(sourceVertex.position, oldDestinationVertex.position));
	insert(graph->quadtree, edgeIndex, Line2D(sourceVertex.position, splitVertex.position));
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
	insert(graph->quadtree, graph->numEdges, Line2D(splitVertex.position, oldDestinationVertex.position));
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
		Line2D edgeLine(sourceVertex.position, destinationVertex.position);

		if (newEdgeLine.intersects(edgeLine, intersection))
		{
			float distance = vml_distance(newEdgeLine.getStart(), intersection);

			if (distance < closestIntersectionDistance)
			{
				if (vml_distance(sourceVertex.position, intersection) <= graph->snapRadius)
				{
					intersectionType = SOURCE;
				}

				else if (vml_distance(destinationVertex.position, intersection) <= graph->snapRadius)
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
		Line2D edgeLine(sourceVertex.position, destinationVertex.position);
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
		Line2D edgeLine(sourceVertex.position, destinationVertex.position);

		if (newEdgeLine.intersects(edgeLine, intersection))
		{
			float distance = vml_distance(start, intersection);

			if (distance < closestIntersectionDistance)
			{
				if (vml_distance(sourceVertex.position, intersection) <= graph->snapRadius)
				{
					intersectionType = SOURCE;
				}

				else if (vml_distance(destinationVertex.position, intersection) <= graph->snapRadius)
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
		Line2D edgeLine(sourceVertex.position, destinationVertex.position);
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
#ifdef USE_QUADTREE
	unsigned int queryResultBufferMemory = graph->maxResultsPerQuery * sizeof(EdgeIndex);
	return (verticesBufferMemory + edgesBufferMemory + queryResultBufferMemory + getAllocatedMemory(graph->quadtree));
#else
	return (verticesBufferMemory + edgesBufferMemory);
#endif
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
#ifdef USE_QUADTREE
	unsigned int queryResultBufferMemory = graph->maxResultsPerQuery * sizeof(EdgeIndex);
	return (verticesBufferMemoryInUse + edgesBufferMemoryInUse + queryResultBufferMemory + getMemoryInUse(graph->quadtree));
#else
	return (verticesBufferMemoryInUse + edgesBufferMemoryInUse);
#endif
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned long getNumCollisionChecks(Graph* graph)
{
#ifdef USE_QUADTREE
	return graph->numCollisionChecks + getNumCollisionChecks(graph->quadtree);
#else
	return graph->numCollisionChecks;
#endif
}

#endif

}
