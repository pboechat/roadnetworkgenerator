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
struct QuadTree;
//////////////////////////////////////////////////////////////////////////
struct GraphTraversal;

//////////////////////////////////////////////////////////////////////////
enum IntersectionType
{
	NONE,
	SOURCE,
	DESTINATION,
	EDGE
};

//////////////////////////////////////////////////////////////////////////
struct Graph : public BaseGraph
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
void initializeGraph(Graph* graph, float snapRadius, unsigned int maxVertices, unsigned int maxEdges, Vertex* vertices, Edge* edges, QuadTree* quadtree, unsigned int maxResultsPerQuery, EdgeIndex* queryResult);
#else
void initializeGraph(Graph* graph, float snapRadius, unsigned int maxVertices, unsigned int maxEdges, Vertex* vertices, Edge* edges);
#endif
//////////////////////////////////////////////////////////////////////////
void copy(Graph* graph, BaseGraph* other);
//////////////////////////////////////////////////////////////////////////
vml_vec2 getPosition(Graph* graph, VertexIndex vertexIndex);
//////////////////////////////////////////////////////////////////////////
VertexIndex createVertex(Graph* graph, const vml_vec2& position);
//////////////////////////////////////////////////////////////////////////
bool addRoad(Graph* graph, VertexIndex sourceIndex, const vml_vec2& direction, VertexIndex& newVertexIndex, vml_vec2& end, bool highway);
//////////////////////////////////////////////////////////////////////////
bool connect(Graph* graph, VertexIndex source, VertexIndex destination, bool highway);
//////////////////////////////////////////////////////////////////////////
void removeDeadEndRoads(Graph* graph);
//////////////////////////////////////////////////////////////////////////
void traverse(const Graph* graph, GraphTraversal& traversal);

#ifdef _DEBUG
//////////////////////////////////////////////////////////////////////////
unsigned int getAllocatedVertices(Graph* graph);
//////////////////////////////////////////////////////////////////////////
unsigned int getVerticesInUse(Graph* graph);
//////////////////////////////////////////////////////////////////////////
unsigned int getAllocatedEdges(Graph* graph);
//////////////////////////////////////////////////////////////////////////
unsigned int getEdgesInUse(Graph* graph);
//////////////////////////////////////////////////////////////////////////
unsigned int getMaxVertexInConnections(Graph* graph);
//////////////////////////////////////////////////////////////////////////
unsigned int getMaxVertexInConnectionsInUse(Graph* graph);
//////////////////////////////////////////////////////////////////////////
unsigned int getMaxVertexOutConnections(Graph* graph);
//////////////////////////////////////////////////////////////////////////
unsigned int getMaxVertexOutConnectionsInUse(Graph* graph);
//////////////////////////////////////////////////////////////////////////
unsigned int getAverageVertexInConnectionsInUse(Graph* graph);
//////////////////////////////////////////////////////////////////////////
unsigned int getAverageVertexOutConnectionsInUse(Graph* graph);
//////////////////////////////////////////////////////////////////////////
unsigned int getAllocatedMemory(Graph* graph);
//////////////////////////////////////////////////////////////////////////
unsigned int getMemoryInUse(Graph* graph);
//////////////////////////////////////////////////////////////////////////
unsigned long getNumCollisionChecks(Graph* graph);
#endif

}

#endif