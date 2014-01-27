#ifndef ROADNETWORKGRAPH_GRAPH_H
#define ROADNETWORKGRAPH_GRAPH_H

#include "Defines.h"
#include <BaseGraph.h>

#include <Box2D.h>
#include <Line2D.h>
#include <Circle2D.h>

#include <vector_math.h>

namespace RoadNetworkGraph
{

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE struct QuadTree;
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE struct GraphTraversal;

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE enum IntersectionType
{
	NONE,
	SOURCE,
	DESTINATION,
	EDGE
};

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE struct Graph : public BaseGraph
{
	unsigned int maxVertices;
	unsigned int maxEdges;
	unsigned int maxResultsPerQuery;
	float snapRadius;
#ifdef _DEBUG
	unsigned long numCollisionChecks;
#endif
#ifdef USE_QUADTREE
	QuadTree* quadtree;
	EdgeIndex* queryResult;
#endif

};

//////////////////////////////////////////////////////////////////////////
#ifdef USE_QUADTREE
GLOBAL_CODE void initializeGraph(Graph* graph, float snapRadius, unsigned int maxVertices, unsigned int maxEdges, Vertex* vertices, Edge* edges, QuadTree* quadtree, unsigned int maxResultsPerQuery, EdgeIndex* queryResult);
#else
GLOBAL_CODE void initializeGraph(Graph* graph, float snapRadius, unsigned int maxVertices, unsigned int maxEdges, Vertex* vertices, Edge* edges);
#endif
//////////////////////////////////////////////////////////////////////////
HOST_CODE void copy(Graph* graph, BaseGraph* other);
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE vml_vec2 getPosition(Graph* graph, VertexIndex vertexIndex);
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE VertexIndex createVertex(Graph* graph, const vml_vec2& position);
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE bool addRoad(Graph* graph, VertexIndex sourceIndex, const vml_vec2& direction, VertexIndex& newVertexIndex, vml_vec2& end, bool highway);
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE bool connect(Graph* graph, VertexIndex source, VertexIndex destination, bool highway);
//////////////////////////////////////////////////////////////////////////
HOST_CODE void removeDeadEndRoads(Graph* graph);
//////////////////////////////////////////////////////////////////////////
HOST_CODE void traverse(const Graph* graph, GraphTraversal& traversal);

#ifdef _DEBUG
//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getAllocatedVertices(Graph* graph);
//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getVerticesInUse(Graph* graph);
//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getAllocatedEdges(Graph* graph);
//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getEdgesInUse(Graph* graph);
//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getMaxVertexInConnections(Graph* graph);
//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getMaxVertexInConnectionsInUse(Graph* graph);
//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getMaxVertexOutConnections(Graph* graph);
//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getMaxVertexOutConnectionsInUse(Graph* graph);
//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getAverageVertexInConnectionsInUse(Graph* graph);
//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getAverageVertexOutConnectionsInUse(Graph* graph);
//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getAllocatedMemory(Graph* graph);
//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getMemoryInUse(Graph* graph);
//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned long getNumCollisionChecks(Graph* graph);
#endif

}

#endif