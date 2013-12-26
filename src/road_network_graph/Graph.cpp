#include <Graph.h>
#include <GraphTraversal.h>

#include <cfloat>

namespace RoadNetworkGraph
{

#ifdef USE_QUADTREE
Graph::Graph(const Box2D& worldBounds, unsigned int quadtreeDepth, float snapRadius, unsigned int maxVertices, unsigned int maxEdges, unsigned int maxResultsPerQuery) :
	vertices(0),
	edges(0),
	lastVertexIndex(0),
	lastEdgeIndex(0),
	snapRadius(snapRadius),
	maxVertices(maxVertices),
	maxEdges(maxEdges),
	maxResultsPerQuery(maxResultsPerQuery),
	quadtree(worldBounds, quadtreeDepth, maxResultsPerQuery),
	queryResult(0)
#ifdef _DEBUG
	, numCollisionChecks(0)
#endif
{
	vertices = new Vertex[maxVertices];
	edges = new Edge[maxEdges];
	queryResult = new EdgeIndex[maxResultsPerQuery];
}
#else
Graph::Graph(const Box2D& worldBounds, float snapRadius, unsigned int maxVertices, unsigned int maxEdges, unsigned int maxResultsPerQuery) :
	vertices(0),
	edges(0),
	lastVertexIndex(0),
	lastEdgeIndex(0),
	snapRadius(snapRadius),
	maxVertices(maxVertices),
	maxEdges(maxEdges),
	maxResultsPerQuery(maxResultsPerQuery)
#ifdef _DEBUG
	, numCollisionChecks(0)
#endif
{
	vertices = new Vertex[maxVertices];
	edges = new Edge[maxEdges];
}
#endif

Graph::~Graph()
{
	if (vertices != 0)
	{
		delete [] vertices;
	}

	if (edges != 0)
	{
		delete[] edges;
	}

#ifdef USE_QUADTREE

	if (queryResult != 0)
	{
		delete[] queryResult;
	}

#endif
}

#ifdef USE_QUADTREE
bool Graph::addRoad(VertexIndex source, const vml_vec2& direction, VertexIndex& newVertex, vml_vec2& end, bool highway)
{
	vml_vec2 start = getPosition(source);
	end = start + direction;
	Line2D newEdgeLine(start, end);
	Box2D newEdgeBounds(newEdgeLine);
	unsigned int querySize;
	quadtree.query(newEdgeBounds, queryResult, querySize);
	EdgeIndex edgeIndex;
	vml_vec2 snapping;
	vml_vec2 intersection;
	IntersectionType intersectionType;

	if (checkIntersection(newEdgeLine, querySize, source, edgeIndex, intersection, intersectionType))
	{
		end = intersection;

		if (intersectionType == SOURCE)
		{
			newVertex = edges[edgeIndex].source;
		}

		else if (intersectionType == DESTINATION)
		{
			newVertex = edges[edgeIndex].destination;
		}

		else if (intersectionType == EDGE)
		{
			newVertex = createVertex(end);
			splitEdge(edgeIndex, newVertex);
		}

		else
		{
			// FIXME: checking invariants
			throw std::exception("unknown intersection type");
		}

		connect(source, newVertex, highway);
		return true;
	}

	else
	{
		Circle2D snapCircle(end, snapRadius);
		quadtree.query(snapCircle, queryResult, querySize);

		if (checkSnapping(snapCircle, querySize, source, snapping, edgeIndex))
		{
			end = snapping;
			newVertex = createVertex(end);
			splitEdge(edgeIndex, newVertex);
			connect(source, newVertex, highway);
			return true;
		}

		else
		{
			newVertex = createVertex(end);
			connect(source, newVertex, highway);
			return false;
		}
	}
}

bool Graph::checkIntersection(const Line2D& newEdgeLine, unsigned int querySize, VertexIndex source, EdgeIndex& edgeIndex, vml_vec2& closestIntersection, IntersectionType& intersectionType)
{
	float closestIntersectionDistance = FLT_MAX;
	edgeIndex = -1;
	intersectionType = NONE;

	for (unsigned int i = 0; i < querySize; i++)
	{
		EdgeIndex queryEdgeIndex = queryResult[i];
		Edge& edge = edges[queryEdgeIndex];

		// avoid intersecting parent or sibling
		if (edge.destination == source || edge.source == source)
		{
			continue;
		}

		Vertex& sourceVertex = vertices[edge.source];
		Vertex& destinationVertex = vertices[edge.destination];
		vml_vec2 intersection;
		Line2D edgeLine(sourceVertex.position, destinationVertex.position);

		if (newEdgeLine.intersects(edgeLine, intersection))
		{
			float distance = vml_distance(newEdgeLine.start, intersection);

			if (distance < closestIntersectionDistance)
			{
				if (vml_distance(sourceVertex.position, intersection) <= snapRadius)
				{
					intersectionType = SOURCE;
				}

				else if (vml_distance(destinationVertex.position, intersection) <= snapRadius)
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
		numCollisionChecks++;
#endif
	}

	return (intersectionType != NONE);
}

bool Graph::checkSnapping(const Circle2D& snapCircle, unsigned int querySize, VertexIndex source, vml_vec2& closestSnapping, EdgeIndex& edgeIndex)
{
	edgeIndex = -1;
	float closestSnappingDistance = FLT_MAX;

	// check for snapping
	for (unsigned int i = 0; i < querySize; i++)
	{
		EdgeIndex queryEdgeIndex = queryResult[i];
		Edge& edge = edges[queryEdgeIndex];

		// avoid snapping parent or sibling
		if (edge.destination == source || edge.source == source)
		{
			continue;
		}

		Vertex& sourceVertex = vertices[edge.source];
		Vertex& destinationVertex = vertices[edge.destination];
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
				throw std::exception("invalid intersection mask");
			}

			if (distance < closestSnappingDistance)
			{
				closestSnappingDistance = distance;
				closestSnapping = snapping;
				edgeIndex = queryEdgeIndex;
			}
		}

#ifdef _DEBUG
		numCollisionChecks++;
#endif
	}

	return (edgeIndex != -1);
}
#else
bool Graph::addRoad(VertexIndex source, const vml_vec2& direction, VertexIndex& newVertex, vml_vec2& end, float& length, bool highway)
{
	vml_vec2 start = getPosition(source);
	end = start + direction;
	EdgeIndex edgeIndex;
	vml_vec2 snapping;
	vml_vec2 intersection;
	IntersectionType intersectionType;

	if (checkIntersection(start, end, source, edgeIndex, intersection, intersectionType))
	{
		end = intersection;
		length = vml_distance(start, end);

		if (intersectionType == SOURCE)
		{
			newVertex = edges[edgeIndex].source;
		}

		else if (intersectionType == DESTINATION)
		{
			newVertex = edges[edgeIndex].destination;
		}

		else if (intersectionType == EDGE)
		{
			newVertex = createVertex(end);
			splitEdge(edgeIndex, newVertex);
		}

		else
		{
			// FIXME: checking invariants
			throw std::exception("unknown intersection type");
		}

		connect(source, newVertex, highway);
		return true;
	}

	else if (checkSnapping(end, source, snapping, edgeIndex))
	{
		end = snapping;
		length = vml_distance(start, end);
		newVertex = createVertex(end);
		splitEdge(edgeIndex, newVertex);
		connect(source, newVertex, highway);
		return true;
	}

	else
	{
		length = vml_distance(start, end);
		newVertex = createVertex(end);
		connect(source, newVertex, highway);
		return false;
	}
}

bool Graph::checkIntersection(const vml_vec2& start, const vml_vec2& end, VertexIndex source, EdgeIndex& edgeIndex, vml_vec2& closestIntersection, IntersectionType& intersectionType)
{
	// check for intersections
	float closestIntersectionDistance = FLT_MAX;
	edgeIndex = -1;
	intersectionType = NONE;
	Line2D newEdgeLine(start, end);

	for (int i = 0; i < lastEdgeIndex; i++)
	{
		Edge& edge = edges[i];

		// avoid intersecting parent or sibling
		if (edge.destination == source || edge.source == source)
		{
			continue;
		}

		Vertex& sourceVertex = vertices[edge.source];
		Vertex& destinationVertex = vertices[edge.destination];
		vml_vec2 intersection;
		Line2D edgeLine(sourceVertex.position, destinationVertex.position);

		if (newEdgeLine.intersects(edgeLine, intersection))
		{
			float distance = vml_distance(start, intersection);

			if (distance < closestIntersectionDistance)
			{
				if (vml_distance(sourceVertex.position, intersection) <= snapRadius)
				{
					intersectionType = SOURCE;
				}

				else if (vml_distance(destinationVertex.position, intersection) <= snapRadius)
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
		numCollisionChecks++;
#endif
	}

	return (intersectionType != NONE);
}

bool Graph::checkSnapping(const vml_vec2& end, VertexIndex source, vml_vec2& closestSnapping, EdgeIndex& edgeIndex)
{
	float closestSnappingDistance = FLT_MAX;
	Circle2D snapCircle(end, snapRadius);

	// check for snapping
	for (int i = 0; i < lastEdgeIndex; i++)
	{
		Edge& edge = edges[i];

		// avoid snapping parent or sibling
		if (edge.destination == source || edge.source == source)
		{
			continue;
		}

		Vertex& sourceVertex = vertices[edge.source];
		Vertex& destinationVertex = vertices[edge.destination];
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
				distance = vml_distance(end, intersection1);
				snapping = intersection1;
			}

			else if (intersectionMask == 2)
			{
				distance = vml_distance(end, intersection2);
				snapping = intersection2;
			}

			else if (intersectionMask == 3)
			{
				float distance1 = vml_distance(end, intersection1);
				float distance2 = vml_distance(end, intersection2);

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
				throw std::exception("invalid intersection mask");
			}

			if (distance < closestSnappingDistance)
			{
				closestSnappingDistance = distance;
				closestSnapping = snapping;
				edgeIndex = i;
			}
		}

#ifdef _DEBUG
		numCollisionChecks++;
#endif
	}

	return (edgeIndex != -1);
}
#endif

unsigned int Graph::getValency(const Vertex& vertex) const
{
	unsigned int valency = 0;

	for (unsigned int i = 0; i < vertex.lastInIndex; i++)
	{
		const Edge& edge = edges[vertex.ins[i]];

		// FIXME: checking invariants
		if (edge.destination != vertex.index)
		{
			throw std::exception("edge.destination != vertex.index");
		}

		const Vertex& source = vertices[edge.source];

		if (source.removed)
		{
			continue;
		}

		valency++;
	}

	for (unsigned int i = 0; i < vertex.lastOutIndex; i++)
	{
		const Edge& edge = edges[vertex.outs[i]];

		// FIXME: checking invariants
		if (edge.source != vertex.index)
		{
			throw std::exception("edge.source != vertex.index");
		}

		const Vertex& destination = vertices[edge.destination];

		if (destination.removed)
		{
			continue;
		}

		valency++;
	}

	return valency;
}

void Graph::removeDeadEndRoads()
{
	bool changed;

	do
	{
		changed = false;

		for (int i = 0; i < lastVertexIndex; i++)
		{
			Vertex& vertex = vertices[i];
			unsigned int valency = getValency(vertex);

			if (vertex.removed)
			{
				continue;
			}

			if (valency == 1)
			{
				vertex.removed = true;
				changed = true;
			}
		}
	}
	while (changed);
}

void Graph::traverse(GraphTraversal& traversal) const
{
	for (int i = 0; i < lastEdgeIndex; i++)
	{
		const Edge& edge = edges[i];
		const Vertex& sourceVertex = vertices[edge.source];
		const Vertex& destinationVertex = vertices[edge.destination];

		if (destinationVertex.removed || destinationVertex.removed)
		{
			continue;
		}

		if (!traversal(sourceVertex.position, destinationVertex.position, edge.highway))
		{
			break;
		}
	}
}

void Graph::connect(VertexIndex source, VertexIndex destination, bool highway)
{
	// FIXME: checking boundaries
	if (lastEdgeIndex >= (int)maxEdges)
	{
		throw std::exception("max. edges overflow");
	}

	Edge& newEdge = edges[lastEdgeIndex];
	newEdge.source = source;
	newEdge.destination = destination;
	newEdge.highway = highway;
	Vertex& sourceVertex = vertices[source];
	Vertex& destinationVertex = vertices[destination];

	// FIXME: checking boundaries
	if (sourceVertex.lastOutIndex >= MAX_VERTEX_OUT_CONNECTIONS)
	{
		throw std::exception("max. vertex connections (out) overflow");
	}

	sourceVertex.outs[sourceVertex.lastOutIndex++] = lastEdgeIndex;

	// FIXME: checking boundaries
	if (destinationVertex.lastInIndex >= MAX_VERTEX_IN_CONNECTIONS)
	{
		throw std::exception("max. vertex connections (in) overflow");
	}

	destinationVertex.ins[destinationVertex.lastInIndex++] = lastEdgeIndex;
#ifdef USE_QUADTREE
	quadtree.insert(lastEdgeIndex, Line2D(sourceVertex.position, destinationVertex.position));
#endif
	lastEdgeIndex++;
}

VertexIndex Graph::createVertex(const vml_vec2& position)
{
	// FIXME: checking boundaries
	if (lastVertexIndex >= (int)maxVertices)
	{
		throw std::exception("max. vertices overflow");
	}

	Vertex& newVertex = vertices[lastVertexIndex];
	newVertex.index = lastVertexIndex;
	newVertex.position = position;
	return lastVertexIndex++;
}

void Graph::splitEdge(EdgeIndex edgeIndex, VertexIndex split)
{
	Edge& splitEdge = edges[edgeIndex];
	VertexIndex oldDestination = splitEdge.destination;
	splitEdge.destination = split;
	Vertex& sourceVertex = vertices[splitEdge.source];
	Vertex& destinationVertex = vertices[oldDestination];
	Vertex& splitVertex = vertices[split];
#ifdef USE_QUADTREE
	quadtree.remove(edgeIndex, Line2D(sourceVertex.position, destinationVertex.position));
	quadtree.insert(edgeIndex, Line2D(sourceVertex.position, splitVertex.position));
#endif

	// FIXME: checking boundaries
	if (splitVertex.lastInIndex >= MAX_VERTEX_IN_CONNECTIONS)
	{
		throw std::exception("max. vertex connections (in) overflow");
	}

	splitVertex.ins[splitVertex.lastInIndex++] = edgeIndex;

	// FIXME: checking boundaries
	if (lastEdgeIndex >= (int)maxEdges)
	{
		throw std::exception("max. edges overflow");
	}

	Edge& newEdge = edges[lastEdgeIndex];
	newEdge.source = split;
	newEdge.destination = oldDestination;
	newEdge.highway = splitEdge.highway;

	// FIXME: checking boundaries
	if (splitVertex.lastOutIndex >= MAX_VERTEX_OUT_CONNECTIONS)
	{
		throw std::exception("max. vertex connections (out) overflow");
	}

	splitVertex.outs[splitVertex.lastOutIndex++] = lastEdgeIndex;
	bool found = false;

	for (unsigned int i = 0; i < destinationVertex.lastInIndex; i++)
	{
		if (destinationVertex.ins[i] == edgeIndex)
		{
			destinationVertex.ins[i] = lastEdgeIndex;
			found = true;
			break;
		}
	}

	// FIXME: checking invariants
	if (!found)
	{
		throw std::exception("!found");
	}

#ifdef USE_QUADTREE
	quadtree.insert(lastEdgeIndex, Line2D(splitVertex.position, destinationVertex.position));
#endif
	lastEdgeIndex++;
}

#ifdef _DEBUG
unsigned int Graph::getAllocatedVertices() const
{
	return maxVertices;
}

unsigned int Graph::getVerticesInUse() const
{
	return lastVertexIndex;
}

unsigned int Graph::getAllocatedEdges() const
{
	return maxEdges;
}

unsigned int Graph::getEdgesInUse() const
{
	return lastEdgeIndex;
}

unsigned int Graph::getMaxVertexInConnections() const
{
	return MAX_VERTEX_IN_CONNECTIONS;
}

unsigned int Graph::getMaxVertexInConnectionsInUse() const
{
	unsigned int maxVerticesInConnectionsInUse = 0;

	for (int i = 0; i < lastVertexIndex; i++)
	{
		Vertex& vertex = vertices[i];

		if (vertex.lastInIndex > maxVerticesInConnectionsInUse)
		{
			maxVerticesInConnectionsInUse = vertex.lastInIndex;
		}
	}

	return maxVerticesInConnectionsInUse;
}

unsigned int  Graph::getMaxVertexOutConnections() const
{
	return MAX_VERTEX_OUT_CONNECTIONS;
}

unsigned int  Graph::getMaxVertexOutConnectionsInUse() const
{
	unsigned int maxVerticesOutConnectionsInUse = 0;

	for (int i = 0; i < lastVertexIndex; i++)
	{
		Vertex& vertex = vertices[i];

		if (vertex.lastOutIndex > maxVerticesOutConnectionsInUse)
		{
			maxVerticesOutConnectionsInUse = vertex.lastOutIndex;
		}
	}

	return maxVerticesOutConnectionsInUse;
}

unsigned int Graph::getAverageVertexInConnectionsInUse() const
{
	unsigned int totalVerticesInConnections = 0;

	for (int i = 0; i < lastVertexIndex; i++)
	{
		Vertex& vertex = vertices[i];
		totalVerticesInConnections += vertex.lastInIndex;
	}

	return totalVerticesInConnections / lastVertexIndex;
}

unsigned int Graph::getAverageVertexOutConnectionsInUse() const
{
	unsigned int totalVerticesOutConnections = 0;

	for (int i = 0; i < lastVertexIndex; i++)
	{
		Vertex& vertex = vertices[i];
		totalVerticesOutConnections += vertex.lastInIndex;
	}

	return totalVerticesOutConnections / lastVertexIndex;
}

unsigned int Graph::getAllocatedMemory() const
{
	unsigned int verticesBufferMemory = maxVertices * sizeof(Vertex);
	unsigned int edgesBufferMemory = maxEdges * sizeof(Vertex);
#ifdef USE_QUADTREE
	unsigned int queryResultBufferMemory = maxResultsPerQuery * sizeof(EdgeIndex);
	unsigned int quadtreeMemory = quadtree.getAllocatedMemory();
	return (verticesBufferMemory + edgesBufferMemory + queryResultBufferMemory + quadtreeMemory);
#else
	return (verticesBufferMemory + edgesBufferMemory);
#endif
}

unsigned int Graph::getMemoryInUse() const
{
	unsigned int verticesBufferMemoryInUse = 0;

	for (int i = 0; i < lastVertexIndex; i++)
	{
		Vertex& vertex = vertices[i];
		// FIXME:
		verticesBufferMemoryInUse += vertex.lastInIndex * sizeof(EdgeIndex) + vertex.lastOutIndex * sizeof(EdgeIndex) + sizeof(vml_vec2) + 2 * sizeof(unsigned int) + sizeof(bool);
	}

	unsigned int edgesBufferMemoryInUse = lastEdgeIndex * sizeof(Vertex);
#ifdef USE_QUADTREE
	unsigned int queryResultBufferMemory = maxResultsPerQuery * sizeof(EdgeIndex);
	unsigned int quadtreeMemoryInUse = quadtree.getMemoryInUse();
	return (verticesBufferMemoryInUse + edgesBufferMemoryInUse + queryResultBufferMemory + quadtreeMemoryInUse);
#else
	return (verticesBufferMemoryInUse + edgesBufferMemoryInUse);
#endif
}

unsigned long Graph::getNumCollisionChecks() const
{
#ifdef USE_QUADTREE
	return numCollisionChecks + quadtree.getNumCollisionChecks();
#else
	return numCollisionChecks;
#endif
}

#ifdef USE_QUADTREE
unsigned int Graph::getMaxEdgesPerQuadrant() const
{
	return MAX_EDGES_PER_QUADRANT;
}

unsigned int Graph::getMaxEdgesPerQuadrantInUse() const
{
	return quadtree.getMaxEdgesPerQuadrantInUse();
}
#endif

#endif

}

