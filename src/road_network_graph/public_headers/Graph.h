#ifndef ROADNETWORKGRAPH_GRAPH_H
#define ROADNETWORKGRAPH_GRAPH_H

#include <Defines.h>
#include <Vertex.h>
#include <Edge.h>
#include <QuadTree.h>

#include <AABB.h>
#include <Line.h>
#include <Circle.h>

#include <glm/glm.hpp>

#include <vector>
#include <exception>

namespace RoadNetworkGraph
{

struct GraphTraversal;

class Graph
{
public:
#ifdef USE_QUADTREE
	Graph(const AABB& worldBounds, unsigned int quadtreeDepth, float snapRadius, unsigned int maxVertices, unsigned int maxEdges, unsigned int maxResultsPerQuery);
#else
	Graph(const AABB& worldBounds, float snapRadius, unsigned int maxVertices, unsigned int maxEdges, unsigned int maxResultsPerQuery);
#endif
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
	bool addRoad(VertexIndex source, const glm::vec3& direction, VertexIndex& newVertex, glm::vec3& end, bool highway);
	void removeDeadEndRoads();
	void traverse(GraphTraversal& traversal) const;
#ifdef _DEBUG
	unsigned int getAllocatedVertices() const;
	unsigned int getVerticesInUse() const;
	unsigned int getAllocatedEdges() const;
	unsigned int getEdgesInUse() const;
	unsigned int getMaxVertexInConnections() const;
	unsigned int getMaxVertexInConnectionsInUse() const;
	unsigned int getMaxVertexOutConnections() const;
	unsigned int getMaxVertexOutConnectionsInUse() const;
	unsigned int getAverageVertexInConnectionsInUse() const;
	unsigned int getAverageVertexOutConnectionsInUse() const;
	unsigned int getAllocatedMemory() const;
	unsigned int getMemoryInUse() const;
	unsigned long getNumCollisionChecks() const;
#ifdef USE_QUADTREE
	unsigned int getMaxEdgesPerQuadrant() const;
	unsigned int getMaxEdgesPerQuadrantInUse() const;
#endif
#endif

private:
	unsigned int maxVertices;
	unsigned int maxEdges;
	unsigned int maxResultsPerQuery;
	Vertex* vertices;
	Edge* edges;
	VertexIndex lastVertexIndex;
	EdgeIndex lastEdgeIndex;
	float snapRadius;
#ifdef _DEBUG
	unsigned long numCollisionChecks;
#endif
#ifdef USE_QUADTREE
	QuadTree quadtree;
	EdgeIndex* queryResult;
#endif

	enum IntersectionType
	{
		NONE,
		SOURCE,
		DESTINATION,
		EDGE
	};

	void connect(VertexIndex source, VertexIndex destination, bool highway);
	void splitEdge(EdgeIndex edge, VertexIndex vertex);
#ifdef USE_QUADTREE
	bool checkIntersection(const Line& newEdgeLine, unsigned int querySize, VertexIndex source, EdgeIndex& edgeIndex, glm::vec3& closestIntersection, IntersectionType& intersectionType);
	bool checkSnapping(const Circle& snapCircle, unsigned int querySize, VertexIndex source, glm::vec3& closestSnapping, EdgeIndex& edgeIndex);
#else
	bool checkIntersection(const glm::vec3& start, const glm::vec3& end, VertexIndex source, EdgeIndex& edgeIndex, glm::vec3& closestIntersection, IntersectionType& intersectionType);
	bool checkSnapping(const glm::vec3& end, VertexIndex source, glm::vec3& closestSnapping, EdgeIndex& edgeIndex);
#endif
	unsigned int getValency(const Vertex& vertex) const;

};

}

#endif