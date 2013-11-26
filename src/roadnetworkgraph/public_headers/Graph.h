#ifndef GRAPH_H
#define GRAPH_H

#include <Defines.h>
#include <Vertex.h>
#include <Edge.h>
#include <QuadTree.h>

#include <AABB.h>
#include <glm/glm.hpp>

#include <vector>
#include <exception>

namespace RoadNetworkGraph
{

struct GraphTraversal;

class Graph
{
public:
	Graph(const AABB& worldBounds, float quadtreeCellArea, float snapRadius);
	~Graph();

	inline glm::vec3 getPosition(VertexIndex vertexIndex) const
	{
		// FIXME: checking invariants
		if (vertexIndex >= lastVertexIndex)
		{
			throw std::exception("invalid vertexIndex");
		}

		return vertices[vertexIndex].position;
	}

	VertexIndex createVertex(const glm::vec3& position);
	bool addRoad(VertexIndex source, const glm::vec3& direction, VertexIndex& newVertex, glm::vec3& end, float& length, bool highway);
	void removeDeadEndRoads();
	void traverse(GraphTraversal& traversal) const;

private:
	QuadTree quadtree;
	Vertex* vertices;
	Edge* edges;
	VertexIndex lastVertexIndex;
	EdgeIndex lastEdgeIndex;
	float snapRadius;
	QuadTree::EdgeReference* queryResult;

	enum IntersectionType
	{
		NONE,
		SOURCE,
		DESTINATION,
		EDGE
	};

	void connect(VertexIndex source, VertexIndex destination, bool highway);
	void splitEdge(EdgeIndex edge, VertexIndex vertex);
	bool checkIntersection(glm::vec3 start, glm::vec3 end, VertexIndex source, EdgeIndex& edgeIndex, glm::vec3& intersection, IntersectionType& intersectionType);
	bool checkSnapping(glm::vec3 end, VertexIndex source, glm::vec3& snapping, EdgeIndex &edgeIndex);
	unsigned int getValency(const Vertex& vertex) const;

};

}

#endif