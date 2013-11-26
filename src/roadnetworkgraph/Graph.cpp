#include <Graph.h>
#include <GraphTraversal.h>

namespace RoadNetworkGraph
{

Graph::Graph(const AABB& worldBounds, float quadtreeCellArea, float snapRadius) : 
		vertices(0),
		edges(0),
		queryResult(0),
		quadtree(worldBounds, quadtreeCellArea), 
		lastVertexIndex(0),
		lastEdgeIndex(0),
		snapRadius(snapRadius)
{
	vertices = new Vertex[MAX_VERTICES];
	edges = new Edge[MAX_EDGES];
	queryResult = new QuadTree::EdgeReference[MAX_EDGE_REFERENCIES_PER_QUERY];
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

	EdgeIndex edgeIndex;
	glm::vec3 snapping;
	glm::vec3 intersection;
	IntersectionType intersectionType;
	if (checkIntersection(start, end, source, edgeIndex, intersection, intersectionType))
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

	else if (checkSnapping(end, source, snapping, edgeIndex))
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

bool Graph::checkIntersection(glm::vec3 start, glm::vec3 end, VertexIndex source, EdgeIndex& edgeIndex, glm::vec3& closestIntersection, IntersectionType& intersectionType) 
{
	// check for intersections
	float closestIntersectionDistance = MAX_DISTANCE;
	edgeIndex = -1;
	intersectionType = NONE;
	Line newEdgeLine(start, end);
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

		glm::vec3 intersection;
		Line edgeLine(sourceVertex.position, destinationVertex.position);
		if (newEdgeLine.intersects(edgeLine, intersection)) 
		{
			float distance = glm::distance(start, intersection);

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
				edgeIndex = i;
			}
		}
	}

	return (intersectionType != NONE);
}

bool Graph::checkSnapping(glm::vec3 end, VertexIndex source, glm::vec3& closestSnapping, EdgeIndex& edgeIndex)
{
	float closestSnappingDistance = MAX_DISTANCE;
	Circle snapCircle(end, snapRadius);
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
				distance = glm::distance(end, intersection1);
				snapping = intersection1;
			}
			else if (intersectionMask == 2)
			{
				distance = glm::distance(end, intersection2);
				snapping = intersection2;
			}
			else if (intersectionMask == 3)
			{
				float distance1 = glm::distance(end, intersection1);
				float distance2 = glm::distance(end, intersection2);
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
	// FIXME: checking invariants
	if (lastEdgeIndex == MAX_EDGES)
	{
		throw std::exception("max edges overflow");
	}

	Edge& newEdge = edges[lastEdgeIndex];

	newEdge.source = source;
	newEdge.destination = destination;
	newEdge.highway = highway;

	Vertex& sourceVertex = vertices[source];
	Vertex& destinationVertex = vertices[destination];

	// FIXME: checking invariants
	if (sourceVertex.lastOutIndex == MAX_VERTEX_CONNECTIONS)
	{
		throw std::exception("vertex outs connections overflow");
	}
	sourceVertex.outs[sourceVertex.lastOutIndex++] = lastEdgeIndex;

	// FIXME: checking invariants
	if (destinationVertex.lastInIndex == MAX_VERTEX_CONNECTIONS)
	{
		throw std::exception("vertex ins connections overflow");
	}
	destinationVertex.ins[destinationVertex.lastInIndex++] = lastEdgeIndex;

	quadtree.insert(lastEdgeIndex, source, destination, sourceVertex.position, destinationVertex.position);

	lastEdgeIndex++;
}

VertexIndex Graph::createVertex(const glm::vec3& position)
{
	// FIXME: checking invariants
	if (lastVertexIndex == MAX_VERTICES)
	{
		throw std::exception("max vertices overflow");
	}

	Vertex& newVertex = vertices[lastVertexIndex];
	newVertex.index = lastVertexIndex;
	newVertex.position = position;
	return lastVertexIndex++;
}

void Graph::splitEdge(EdgeIndex edge, VertexIndex split)
{
	Edge& splitEdge = edges[edge];

	VertexIndex oldDestination = splitEdge.destination;
	splitEdge.destination = split;
	Vertex& sourceVertex = vertices[split];

	// FIXME: checking invariants
	if (sourceVertex.lastInIndex == MAX_VERTEX_CONNECTIONS)
	{
		throw std::exception("vertex ins connections overflow");
	}
	sourceVertex.ins[sourceVertex.lastInIndex++] = edge;
	
	// FIXME: checking invariants
	if (lastEdgeIndex == MAX_EDGES)
	{
		throw std::exception("max edges overflow");
	}

	Edge& newEdge = edges[lastEdgeIndex];

	newEdge.source = split;
	newEdge.destination = oldDestination;
	newEdge.highway = splitEdge.highway;
	
	Vertex& destinationVertex = vertices[oldDestination];

	// FIXME: checking invariants
	if (sourceVertex.lastOutIndex == MAX_VERTEX_CONNECTIONS)
	{
		throw std::exception("vertex outs connections overflow");
	}
	sourceVertex.outs[sourceVertex.lastOutIndex++] = lastEdgeIndex;

	bool found = false;
	for (unsigned int i = 0; i < destinationVertex.lastInIndex; i++)
	{
		if (destinationVertex.ins[i] == edge)
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
	
	lastEdgeIndex++;
}

}