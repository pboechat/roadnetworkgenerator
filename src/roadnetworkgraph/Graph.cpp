#include <Graph.h>
#include <GraphTraversal.h>

namespace RoadNetworkGraph
{

enum IntersectionType
{
	NONE,
	SOURCE,
	DESTINATION,
	EDGE
};

Graph::Graph(const AABB& worldBounds, float quadtreeCellArea, float snapRadius) : 
		quadtree(worldBounds, quadtreeCellArea), 
		lastVertexIndex(0),
		lastEdgeIndex(0),
		snapRadius(snapRadius)
{
	glm::vec3 worldSize = worldBounds.getExtents();
	addVertex(-1, glm::vec3(worldSize.x / 2.0f, worldSize.y / 2.0f, 0.0f));
}

Graph::~Graph() {}

bool Graph::addRoad(VertexIndex source, const glm::vec3& direction, VertexIndex& newVertexIndex, glm::vec3& end, bool highway)
{
	glm::vec3 start = getPosition(source);
	end = start + direction;

	// check for intersections
	float closestIntersectionDistance = MAX_DISTANCE;
	EdgeIndex intersectedEdgeIndex = -1;
	glm::vec3 closestIntersection;
	IntersectionType intersectionType = NONE;
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
				if (glm::distance(sourceVertex.position, intersection) <= 0.5f) 
				{
					intersectionType = SOURCE;
				}
				else if (glm::distance(destinationVertex.position, intersection) <= 0.5f)
				{
					intersectionType = DESTINATION;
				}
				else
				{
					intersectionType = EDGE;
				}

				closestIntersectionDistance = distance;
				closestIntersection = intersection;
				intersectedEdgeIndex = i;
			}
		}
	}

	if (intersectionType != NONE)
	{
		end = closestIntersection;
		Edge& intersectedEdge = edges[intersectedEdgeIndex];

		if (intersectionType == SOURCE)
		{
			newVertexIndex = intersectedEdge.source;
		}
		else if (intersectionType == DESTINATION)
		{
			newVertexIndex = intersectedEdge.destination;
		}
		else if (intersectionType == EDGE)
		{
			newVertexIndex = addVertex(source, end);

			VertexIndex oldDestination = intersectedEdge.destination;
			intersectedEdge.destination = newVertexIndex;

			addConnection(newVertexIndex, oldDestination, intersectedEdge.highway);
		}
		else
		{
			// FIXME: checking invariants
			throw std::exception("invalid intersection type");
		}

		addConnection(source, newVertexIndex, highway);

		return true;
	}
	else
	{
		float closestSnappingDistance = MAX_DISTANCE;
		EdgeIndex snappedEdgeIndex = -1;
		glm::vec3 closestSnapping;
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
					throw std::exception("invalid intersection mask: " + intersectionMask);
				}

				if (distance < closestSnappingDistance)
				{
					closestSnappingDistance = distance;
					closestSnapping = snapping;
					snappedEdgeIndex = i;
				}
			}
		}

		if (snappedEdgeIndex != -1)
		{
			end = closestSnapping;
			Edge& snappedEdge = edges[snappedEdgeIndex];

			newVertexIndex = addVertex(source, end);

			VertexIndex oldDestination = snappedEdge.destination;
			snappedEdge.destination = newVertexIndex;

			addConnection(newVertexIndex, oldDestination, snappedEdge.highway);
			addConnection(source, newVertexIndex, highway);

			return true;
		}

		newVertexIndex = addVertex(source, end);
		addConnection(source, newVertexIndex, highway);

		return false;
	}
}

void Graph::removeDeadEndRoads()
{
	for (int i = 1; i < lastVertexIndex; i++) // skip first vertex
	{
		Vertex& vertex = vertices[i];
		if (vertex.lastConnectionIndex == 0) 
		{
			vertex.removed = true;
		}
	}
}

void Graph::traverse(GraphTraversal& traversal) const
{
	for (int i = 0; i < lastEdgeIndex; i++)
	{
		const Edge& edge = edges[i];

		const Vertex& sourceVertex = vertices[edge.source];
		const Vertex& destinationVertex = vertices[edge.destination];

		if (sourceVertex.removed || destinationVertex.removed)
		{
			continue;
		}

		if (!traversal(*this, edge.source, edge.destination, edge.highway)) 
		{
			break;
		}
	}
}

void Graph::addConnection(VertexIndex source, VertexIndex destination, bool highway)
{
	Edge& newEdge = edges[lastEdgeIndex];
	newEdge.source = source;
	newEdge.destination = destination;
	newEdge.highway = highway;
	Vertex& sourceVertex = vertices[source];
	Vertex& destinationVertex = vertices[destination];
	if (sourceVertex.lastConnectionIndex == MAX_VERTEX_CONNECTIONS)
	{
		// FIXME: checking invariants
		throw std::exception("vertex connection overflow");
	}
	sourceVertex.connections[sourceVertex.lastConnectionIndex++] = lastEdgeIndex;
	// REENABLE:
	//quadtree.insert(lastEdgeIndex, source, destination, sourceVertex.position, destinationVertex.position);
	lastEdgeIndex++;
}

VertexIndex Graph::addVertex(VertexIndex source, const glm::vec3& position)
{
	Vertex& newVertex = vertices[lastVertexIndex];
	newVertex.index = lastVertexIndex;
	newVertex.source = source;
	newVertex.position = position;
	return lastVertexIndex++;
}

}