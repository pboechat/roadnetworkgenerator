#ifndef GRAPH_H
#define GRAPH_H

#include <Defines.h>
#include <Vertex.h>
#include <Edge.h>
#include <Quadtree.h>

#include <AABB.h>
#include <glm/glm.hpp>

#include <exception>

namespace RoadNetworkGraph
{

struct GraphTraversal;

class Graph
{
public:
	Graph(const AABB& worldBounds, float quadtreeCellArea, float quadtreeQueryRadius);
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

	bool addRoad(VertexIndex source, const glm::vec3& direction, VertexIndex& newVertexIndex, glm::vec3& end, bool highway);
	void traverse(GraphTraversal& traversal) const;

private:
	QuadTree quadtree;
	Vertex vertices[MAX_VERTICES];
	Edge edges[MAX_EDGES];
	VertexIndex lastVertexIndex;
	EdgeIndex lastEdgeIndex;
	float queryRadius;
	EdgeReference queryResult[MAX_EDGE_REFERENCIES_PER_QUERY];

	void addConnection(VertexIndex source, VertexIndex destination, bool highway);
	VertexIndex addVertex(VertexIndex source, const glm::vec3& position);

};

}

#endif