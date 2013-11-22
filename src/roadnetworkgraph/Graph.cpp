#include <Graph.h>
#include <GraphTraversal.h>

namespace RoadNetworkGraph
{

Graph::Graph(const AABB& worldBounds, float quadtreeCellArea, float quadtreeQueryRadius) : 
		quadtree(worldBounds, quadtreeCellArea), 
		lastVertexIndex(0),
		lastEdgeIndex(0),
		queryRadius(quadtreeQueryRadius)
{
	glm::vec3 worldSize = worldBounds.getExtents();
	addVertex(-1, glm::vec3(worldSize.x / 2.0f, worldSize.y / 2.0f, 0.0f));
}

Graph::~Graph() {}

bool Graph::addRoad(VertexIndex source, const glm::vec3& direction, VertexIndex& newVertexIndex, glm::vec3& end, bool highway)
{
	glm::vec3 start = getPosition(source);
	end = start + direction;

	if (end.x >= 340 && end.x < 360 && end.y >= 890 && end.y < 910)
	{
		int a = 0;
	}

	float shortestDistance = MAX_DISTANCE;
	EdgeIndex intersectedEdgeIndex = -1;
	glm::vec3 closestIntersection;
	Line newEdgeLine(start, end);
	for (int i = 0; i < lastEdgeIndex; i++)
	{
		Edge& edge = edges[i];

		// avoid collision with parent or sibling
		if (edge.destination == source || edge.source == source)
		{
			continue;
		}

		Vertex& sourceVertex = vertices[edge.source];
		Vertex& destinationVertex = vertices[edge.destination];

		Line edgeLine(sourceVertex.position, destinationVertex.position);

		glm::vec3 intersection;
		if (newEdgeLine.intersects(edgeLine, intersection)) 
		{
			float distance = glm::distance(start, intersection);

			if (distance < shortestDistance)
			{
				shortestDistance = distance;
				closestIntersection = intersection;
				intersectedEdgeIndex = i;
			}
		}
	}

	if (intersectedEdgeIndex != -1)
	{
		end = closestIntersection;
		Edge& intersectedEdge = edges[intersectedEdgeIndex];

		newVertexIndex = addVertex(source, end);

		VertexIndex oldDestination = intersectedEdge.destination;
		intersectedEdge.destination = newVertexIndex;

		addConnection(newVertexIndex, oldDestination, intersectedEdge.highway);
		addConnection(source, newVertexIndex, highway);

		return true;
	}
	else
	{
		newVertexIndex = addVertex(source, end);
		addConnection(source, newVertexIndex, highway);
		return false;
	}

	// REENABLE:
	/*unsigned int size;
	quadtree.query(Circle(position, queryRadius), queryResult, size);

	if (size > 0)
	{
		Line newEdge(start, position);
		EdgeIndex intersectedEdgeIndex;
		glm::vec3 closestIntersection;
		float minimalDistance = MAX_DISTANCE;
		IntersectionType closestIntersectionType = NONE;
		for (unsigned int i = 0; i < size; i++)
		{
			EdgeReference& edgeReference = queryResult[i];

			if (edgeReference.destination == source || edgeReference.source == source)
			{
				continue;
			}

			Line edge(edgeReference.sourcePosition, edgeReference.destinationPosition);

			IntersectionType intersectionType = NONE;
			float distance;
			glm::vec3 intersectionPoint;
			if (newEdge.contains(edgeReference.sourcePosition))
			{
				distance = glm::distance(start, edgeReference.sourcePosition);
				intersectionPoint = edgeReference.sourcePosition;
				intersectionType = SOURCE_VERTEX;
			}

			else if (newEdge.contains(edgeReference.destinationPosition))
			{
				distance = glm::distance(start, edgeReference.destinationPosition);
				intersectionPoint = edgeReference.destinationPosition;
				intersectionType = DESTINATION_VERTEX;
			}

			else if (edge.intersects(newEdge, intersectionPoint)) 
			{
				distance = glm::distance(start, intersectionPoint);
				intersectionType = EDGE;
			}

			if (intersectionType != NONE)
			{
				if (distance < minimalDistance)
				{
					minimalDistance = distance;
					closestIntersection = intersectionPoint;
					intersectedEdgeIndex = edgeReference.index;
					closestIntersectionType = intersectionType;
				}
			}
		}

		if (closestIntersectionType != NONE)
		{
			position = closestIntersection;
			Edge& intersectedEdge = edges[intersectedEdgeIndex];

			if (closestIntersectionType == SOURCE_VERTEX)
			{
				newVertexIndex = intersectedEdge.source;
			}

			else if (closestIntersectionType == DESTINATION_VERTEX)
			{
				newVertexIndex = intersectedEdge.destination;
			}

			else if (closestIntersectionType == EDGE)
			{
				newVertexIndex = addVertex(source, position);

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
	}

	newVertexIndex = addVertex(source, position);
	
	if (!addConnection(source, newVertexIndex, highway))
	{
		// FIXME: checking invariants
		throw std::exception("vertex connection overflow");
	}

	return false;*/
}

void Graph::traverse(GraphTraversal& traversal) const
{
	for (int i = 0; i < lastEdgeIndex; i++)
	{
		const Edge& edge = edges[i];
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