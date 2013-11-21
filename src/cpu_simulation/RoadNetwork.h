#ifndef ROADNETWORK_H
#define ROADNETWORK_H

#include <Configuration.h>
#include <AABB.h>
#include <Circle.h>

#include <glm/glm.hpp>

#include <vector>
#include <exception>

namespace RoadNetwork
{

#define MAX_VERTICES 10000
#define MAX_VERTEX_CONNECTIONS 20
// MAX_VERTICES * MAX_VERTEX_CONNECTIONS
#define MAX_EDGES 40000
#define MAX_EDGE_REFERENCIES_PER_QUERY 1000
#define MAX_DISTANCE 10000

typedef int VertexIndex;
typedef int EdgeIndex;

struct Vertex
{
	VertexIndex index;
	VertexIndex source;
	glm::vec3 position;
	EdgeIndex connections[MAX_VERTEX_CONNECTIONS];
	unsigned int lastConnectionIndex;

	Vertex() : lastConnectionIndex(0)
	{
	}

	bool addConnection(EdgeIndex edgeIndex)
	{
		if (lastConnectionIndex == MAX_VERTEX_CONNECTIONS)
		{
			return false;
		}
		connections[lastConnectionIndex++] = edgeIndex;
		return true;
	}

};

struct Edge
{
	bool highway;
	VertexIndex source;
	VertexIndex destination;

};

class Graph;

struct Traversal
{
	virtual bool operator () (const Graph& graph, VertexIndex source, VertexIndex destination, bool highway) = 0;

};

struct EdgeReference
{
	EdgeIndex index;
	glm::vec3 sourcePosition;
	glm::vec3 destinationPosition;

	EdgeReference& operator = (const EdgeReference& other)
	{
		index = other.index;
		sourcePosition = other.sourcePosition;
		destinationPosition = other.destinationPosition;
		return *this;
	}

};

class QuadTree
{
public:
	QuadTree(const AABB& bounds, float cellArea) : bounds(bounds), cellArea(cellArea), lastEdgeReferenceIndex(0), northWest(0), northEast(0), southWest(0), southEast(0) {}
	~QuadTree()
	{
		if (northWest != 0)
		{
			delete northWest;
		}

		if (northEast != 0)
		{
			delete northEast;
		}

		if (southWest != 0)
		{
			delete southWest;
		}

		if (southEast != 0)
		{
			delete southEast;
		}
	}

	bool insert(EdgeIndex index, const glm::vec3& sourcePosition, const glm::vec3& destinationPosition)
	{
		if (!bounds.contains(sourcePosition) && !bounds.contains(destinationPosition))
		{
			return false;
		}

		if (bounds.getArea() <= cellArea)
		{
			addEdgeReference(index, sourcePosition, destinationPosition);
			return true;
		}

		if (northWest == 0)
		{
			subdivide();
		}

		if (northWest->insert(index, sourcePosition, destinationPosition))
		{
			return true;
		}

		if (northEast->insert(index, sourcePosition, destinationPosition))
		{
			return true;
		}

		if (southWest->insert(index, sourcePosition, destinationPosition))
		{
			return true;
		}

		if (southEast->insert(index, sourcePosition, destinationPosition))
		{
			return true;
		}

		// FIXME: should never happen!
		throw std::exception("couldn't insert point");
	}

	void subdivide()
	{
		float halfWidth = bounds.getExtents().x / 2.0f;
		float halfHeight = bounds.getExtents().y / 2.0f;
		northWest = new QuadTree(AABB(bounds.min.x, bounds.min.y + halfHeight, halfWidth, halfHeight), cellArea);
		northEast = new QuadTree(AABB(bounds.min.x + halfWidth, bounds.min.y + halfHeight, halfWidth, halfHeight), cellArea);
		southWest = new QuadTree(AABB(bounds.min.x, bounds.min.y, halfWidth, halfHeight), cellArea);
		southEast = new QuadTree(AABB(bounds.min.x + halfWidth, bounds.min.y, halfWidth, halfHeight), cellArea);
	}

	void query(const Circle& circle, EdgeReference* edgeReferences, unsigned int& size) const
	{
		size = 0;
		query_(circle, edgeReferences, size);
	}

	inline bool isLeaf() const
	{
		return northEast == 0;
	}

	inline bool notEmpty() const
	{
		return lastEdgeReferenceIndex > 0;
	}

	inline const QuadTree* getNorthWest() const
	{
		return northWest;
	}

	inline const QuadTree* getNorthEast() const
	{
		return northEast;
	}

	inline const QuadTree* getSouthWest() const
	{
		return southWest;
	}

	inline const QuadTree* getSouthEast() const
	{
		return southEast;
	}

	inline const AABB& getBounds() const
	{
		return bounds;
	}

private:
	AABB bounds;
	float cellArea;
	QuadTree* northWest;
	QuadTree* northEast;
	QuadTree* southWest;
	QuadTree* southEast;
	EdgeReference edgeReferences[MAX_VERTICES];
	unsigned int lastEdgeReferenceIndex;

	void query_(const Circle& circle, EdgeReference* edgeReferences, unsigned int& size) const
	{
		if (size == MAX_EDGE_REFERENCIES_PER_QUERY)
		{
			return;
		}

		if (!bounds.intersects(circle))
		{
			return;
		}

		if (isLeaf())
		{
			for (unsigned int i = 0; i < lastEdgeReferenceIndex; i++)
			{
				const EdgeReference& edgeReference = this->edgeReferences[i];

				if (Line(edgeReference.sourcePosition, edgeReference.destinationPosition).intersects(circle))
				{
					edgeReferences[size++] = edgeReference;

					if (size == MAX_EDGE_REFERENCIES_PER_QUERY)
					{
						return;
					}
				}
			}
		}

		else
		{
			northWest->query_(circle, edgeReferences, size);
			northEast->query_(circle, edgeReferences, size);
			southWest->query_(circle, edgeReferences, size);
			southEast->query_(circle, edgeReferences, size);
		}
	}

	void addEdgeReference(EdgeIndex edgeIndex, const glm::vec3& sourcePosition, const glm::vec3& destinationPosition) 
	{
		EdgeReference& edgeReference = edgeReferences[lastEdgeReferenceIndex++];
		edgeReference.index = edgeIndex;
		edgeReference.sourcePosition = sourcePosition;
		edgeReference.destinationPosition = destinationPosition;
	}

};

enum IntersectionType
{
	NONE,
	SOURCE_VERTEX,
	DESTINATION_VERTEX,
	EDGE

};

class Graph
{
public:
	Graph(const Configuration& configuration) : 
		quadtree(AABB(0.0f, 0.0f, (float)configuration.worldWidth, (float)configuration.worldHeight), (float)configuration.quadtreeCellArea), 
		lastVertexIndex(0),
		lastEdgeIndex(0),
		queryRadius((float)configuration.quadtreeQueryRadius)
	{
		addVertex(-1, glm::vec3(configuration.worldWidth / 2.0f, configuration.worldHeight / 2.0f, 0.0f));
	}

	~Graph() {}

	glm::vec3 getPosition(VertexIndex vertexIndex) const
	{
		// FIXME: checking invariants
		if (vertexIndex >= lastVertexIndex)
		{
			throw std::exception("invalid vertexIndex");
		}

		return vertices[vertexIndex].position;
	}

	bool addRoad(VertexIndex source, const glm::vec3& direction, VertexIndex& newVertexIndex, glm::vec3& position, bool highway)
	{
		glm::vec3 start = getPosition(source);
		position = start + direction;

		unsigned int size;
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
					if (position.x >= 900 && position.y <= 1000)
					{
						int a = 0;
					}

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

		if (position.x >= 900 && position.y <= 1000)
		{
			int a = 0;
		}

		newVertexIndex = addVertex(source, position);
		
		if (!addConnection(source, newVertexIndex, highway))
		{
			// FIXME: checking invariants
			throw std::exception("vertex connection overflow");
		}

		return false;
	}

	void traverse(Traversal& traversal) const
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

private:
	QuadTree quadtree;
	Vertex vertices[MAX_VERTICES];
	Edge edges[MAX_EDGES];
	VertexIndex lastVertexIndex;
	EdgeIndex lastEdgeIndex;
	float queryRadius;
	EdgeReference queryResult[MAX_EDGE_REFERENCIES_PER_QUERY];

	bool addConnection(VertexIndex source, VertexIndex destination, bool highway)
	{
		Edge& newEdge = edges[lastEdgeIndex];
		newEdge.source = source;
		newEdge.destination = destination;
		newEdge.highway = highway;
		Vertex& sourceVertex = vertices[source];
		Vertex& destinationVertex = vertices[destination];
		if (sourceVertex.addConnection(lastEdgeIndex))
		{
			quadtree.insert(lastEdgeIndex, sourceVertex.position, destinationVertex.position);
			lastEdgeIndex++;
			return true;
		}

		else
		{
			return false;
		}
	}

	VertexIndex addVertex(VertexIndex source, const glm::vec3& position)
	{
		Vertex& newVertex = vertices[lastVertexIndex];
		newVertex.index = lastVertexIndex;
		newVertex.source = source;
		newVertex.position = position;
		return lastVertexIndex++;
	}

};

}

#endif