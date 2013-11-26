#include <Graph.h>
#include <GraphTraversal.h>

namespace RoadNetworkGraph
{

Graph::Graph(const AABB& worldBounds, unsigned int quadtreeDepth, float snapRadius) : 
		vertices(0),
		edges(0),
		queryResult(0),
		quadtree(worldBounds, quadtreeDepth), 
		lastVertexIndex(0),
		lastEdgeIndex(0),
		snapRadius(snapRadius)
{
	vertices = new Vertex[MAX_VERTICES];
	edges = new Edge[MAX_EDGES];
	queryResult = new EdgeIndex[MAX_RESULTS_PER_QUERY];
}

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

	if (queryResult != 0)
	{
		delete[] queryResult;
	}
}

bool Graph::addRoad(VertexIndex source, const glm::vec3& direction, VertexIndex& newVertex, glm::vec3& end, float& length, bool highway)
{
	glm::vec3 start = getPosition(source);
	end = start + direction;

	Line newEdgeLine(start, end);
	AABB newEdgeBounds(newEdgeLine);

	unsigned int querySize;
	quadtree.query(newEdgeBounds, queryResult, querySize);

	EdgeIndex edgeIndex;
	glm::vec3 snapping;
	glm::vec3 intersection;
	IntersectionType intersectionType;
	if (checkIntersection(newEdgeLine, querySize, source, edgeIndex, intersection, intersectionType))
	{
		end = intersection;
		length = glm::distance(start, end);

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
		Circle snapCircle(end, snapRadius);
		quadtree.query(snapCircle, queryResult, querySize);
		
		if (checkSnapping(snapCircle, querySize, source, snapping, edgeIndex))
		{
			end = snapping;
			length = glm::distance(start, end);

			newVertex = createVertex(end);
			splitEdge(edgeIndex, newVertex);
			connect(source, newVertex, highway);

			return true;
		} 

		else 
		{
			length = glm::distance(start, end);
			newVertex = createVertex(end);
			connect(source, newVertex, highway);

			return false;
		}
	}
}

bool Graph::checkIntersection(const Line& newEdgeLine, unsigned int querySize, VertexIndex source, EdgeIndex& edgeIndex, glm::vec3& closestIntersection, IntersectionType& intersectionType) const
{
	float closestIntersectionDistance = MAX_DISTANCE;
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

		glm::vec3 intersection;
		Line edgeLine(sourceVertex.position, destinationVertex.position);
		if (newEdgeLine.intersects(edgeLine, intersection)) 
		{
			float distance = glm::distance(newEdgeLine.start, intersection);

			if (distance < closestIntersectionDistance)
			{
				if (glm::distance(sourceVertex.position, intersection) <= snapRadius) 
				{
					intersectionType = SOURCE;
				}
				else if (glm::distance(destinationVertex.position, intersection) <= snapRadius)
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
	}

	return (intersectionType != NONE);
}

bool Graph::checkSnapping(const Circle& snapCircle, unsigned int querySize, VertexIndex source, glm::vec3& closestSnapping, EdgeIndex& edgeIndex) const
{
	edgeIndex = -1;
	float closestSnappingDistance = MAX_DISTANCE;
	
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

		Line edgeLine(sourceVertex.position, destinationVertex.position);

		glm::vec3 intersection1;
		glm::vec3 intersection2;
		glm::vec3 snapping;
		int intersectionMask = edgeLine.intersects(snapCircle, intersection1, intersection2);
		if (intersectionMask > 0) 
		{
			float distance;

			if (intersectionMask == 1)
			{
				distance = glm::distance(snapCircle.center, intersection1);
				snapping = intersection1;
			}
			else if (intersectionMask == 2)
			{
				distance = glm::distance(snapCircle.center, intersection2);
				snapping = intersection2;
			}
			else if (intersectionMask == 3)
			{
				float distance1 = glm::distance(snapCircle.center, intersection1);
				float distance2 = glm::distance(snapCircle.center, intersection2);
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
	}

	return (edgeIndex != -1);
}

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
	} while (changed);
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
	if (lastEdgeIndex == MAX_EDGES)
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
	if (sourceVertex.lastOutIndex == MAX_VERTEX_CONNECTIONS)
	{
		throw std::exception("max. vertex connections (out) overflow");
	}
	sourceVertex.outs[sourceVertex.lastOutIndex++] = lastEdgeIndex;

	// FIXME: checking boundaries
	if (destinationVertex.lastInIndex == MAX_VERTEX_CONNECTIONS)
	{
		throw std::exception("max. vertex connections (in) overflow");
	}
	destinationVertex.ins[destinationVertex.lastInIndex++] = lastEdgeIndex;

	quadtree.insert(lastEdgeIndex, Line(sourceVertex.position, destinationVertex.position));

	lastEdgeIndex++;
}

VertexIndex Graph::createVertex(const glm::vec3& position)
{
	// FIXME: checking boundaries
	if (lastVertexIndex == MAX_VERTICES)
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

	quadtree.remove(edgeIndex, Line(sourceVertex.position, destinationVertex.position));
	quadtree.insert(edgeIndex, Line(sourceVertex.position, splitVertex.position));

	// FIXME: checking boundaries
	if (splitVertex.lastInIndex == MAX_VERTEX_CONNECTIONS)
	{
		throw std::exception("max. vertex connections (in) overflow");
	}
	splitVertex.ins[splitVertex.lastInIndex++] = edgeIndex;
	
	// FIXME: checking boundaries
	if (lastEdgeIndex == MAX_EDGES)
	{
		throw std::exception("max. edges overflow");
	}

	Edge& newEdge = edges[lastEdgeIndex];
	newEdge.source = split;
	newEdge.destination = oldDestination;
	newEdge.highway = splitEdge.highway;

	// FIXME: checking boundaries
	if (splitVertex.lastOutIndex == MAX_VERTEX_CONNECTIONS)
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

	quadtree.insert(lastEdgeIndex, Line(splitVertex.position, destinationVertex.position));
	
	lastEdgeIndex++;
}

}