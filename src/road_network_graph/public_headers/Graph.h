#ifndef ROADNETWORKGRAPH_GRAPH_H
#define ROADNETWORKGRAPH_GRAPH_H

#include <Defines.h>
#include <Vertex.h>
#include <Edge.h>
#include <QuadTree.h>

#include <Box2D.h>
#include <Line2D.h>
#include <Circle2D.h>

#include <vector_math.h>

#include <vector>
#include <exception>

namespace RoadNetworkGraph
{

struct GraphTraversal;

class Graph
{
public:
#ifdef USE_QUADTREE
	Graph(const Box2D& worldBounds, unsigned int quadtreeDepth, float snapRadius, unsigned int maxVertices, unsigned int maxEdges, unsigned int maxResultsPerQuery);
#else
	Graph(const Box2D& worldBounds, float snapRadius, unsigned int maxVertices, unsigned int maxEdges, unsigned int maxResultsPerQuery);
#endif
	~Graph();

	inline vml_vec2 getPosition(VertexIndex vertexIndex) const
	{
		// FIXME: checking invariants
		if (vertexIndex >= lastVertexIndex)
		{
			throw std::exception("invalid vertexIndex");
		}

		return vertices[vertexIndex].position;
	}

	VertexIndex createVertex(const vml_vec2& position);
	bool addRoad(VertexIndex source, const vml_vec2& direction, VertexIndex& newVertex, vml_vec2& end, bool highway);
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
	bool checkIntersection(const Line2D& newEdgeLine, unsigned int querySize, VertexIndex source, EdgeIndex& edgeIndex, vml_vec2& closestIntersection, IntersectionType& intersectionType);
	bool checkSnapping(const Circle2D& snapCircle, unsigned int querySize, VertexIndex source, vml_vec2& closestSnapping, EdgeIndex& edgeIndex);
#else
	bool checkIntersection(const vml_vec2& start, const vml_vec2& end, VertexIndex source, EdgeIndex& edgeIndex, vml_vec2& closestIntersection, IntersectionType& intersectionType);
	bool checkSnapping(const vml_vec2& end, VertexIndex source, vml_vec2& closestSnapping, EdgeIndex& edgeIndex);
#endif
	unsigned int getValency(const Vertex& vertex) const;

};

}

#endif