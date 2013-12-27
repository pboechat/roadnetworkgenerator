#include <Globals.h>

#include <memory>

//////////////////////////////////////////////////////////////////////////
Configuration* g_configuration = 0;
//////////////////////////////////////////////////////////////////////////
RoadNetworkGraph::Graph* g_graph = 0;
//////////////////////////////////////////////////////////////////////////
unsigned char* g_populationDensitiesSamplingBuffer = 0;
//////////////////////////////////////////////////////////////////////////
unsigned int* g_distancesSamplingBuffer = 0;
//////////////////////////////////////////////////////////////////////////
RoadNetworkGraph::Vertex* g_vertices = 0;
//////////////////////////////////////////////////////////////////////////
RoadNetworkGraph::Edge* g_edges = 0;
#ifdef USE_QUADTREE
//////////////////////////////////////////////////////////////////////////
extern RoadNetworkGraph::QuadTree* g_quadtree = 0;
//////////////////////////////////////////////////////////////////////////
RoadNetworkGraph::Quadrant* g_quadrants = 0;
//////////////////////////////////////////////////////////////////////////
RoadNetworkGraph::QuadrantEdges* g_quadrantsEdges = 0;
//////////////////////////////////////////////////////////////////////////
RoadNetworkGraph::EdgeIndex* g_queryResults = 0;
#endif

//////////////////////////////////////////////////////////////////////////
void initializeSamplingBuffers()
{
	disposeSamplingBuffers();
	g_populationDensitiesSamplingBuffer = new unsigned char[g_configuration->samplingArc];
	g_distancesSamplingBuffer = new unsigned int[g_configuration->samplingArc];
}

//////////////////////////////////////////////////////////////////////////
void disposeSamplingBuffers()
{
	if (g_populationDensitiesSamplingBuffer != 0)
	{
		delete[] g_populationDensitiesSamplingBuffer;
	}

	if (g_distancesSamplingBuffer != 0)
	{
		delete[] g_distancesSamplingBuffer;
	}
}

//////////////////////////////////////////////////////////////////////////
void initializeGraphBuffers()
{
	disposeGraphBuffers();
	g_vertices = new RoadNetworkGraph::Vertex[g_configuration->maxVertices];
	g_edges = new RoadNetworkGraph::Edge[g_configuration->maxEdges];
	memset(g_vertices, 0, sizeof(RoadNetworkGraph::Vertex) * g_configuration->maxVertices);
	memset(g_edges, 0, sizeof(RoadNetworkGraph::Edge) * g_configuration->maxEdges);
}

//////////////////////////////////////////////////////////////////////////
void disposeGraphBuffers()
{
	if (g_vertices != 0)
	{
		delete[] g_vertices;
	}

	if (g_edges != 0)
	{
		delete[] g_edges;
	}
}

#ifdef USE_QUADTREE
//////////////////////////////////////////////////////////////////////////
void initializeQuadtreeBuffers()
{
	disposeQuadtreeBuffers();
	g_queryResults = new RoadNetworkGraph::EdgeIndex[g_configuration->maxResultsPerQuery];
	// TODO:
	g_quadrants = new RoadNetworkGraph::Quadrant[512];
	g_quadrantsEdges = new RoadNetworkGraph::QuadrantEdges[5000];
	memset(g_quadrants, 0, sizeof(RoadNetworkGraph::Quadrant) * 512);
	memset(g_quadrantsEdges, 0, sizeof(RoadNetworkGraph::QuadrantEdges) * 5000);
}

//////////////////////////////////////////////////////////////////////////
void disposeQuadtreeBuffers()
{
	if (g_quadrants != 0)
	{
		delete[] g_quadrants;
	}

	if (g_quadrantsEdges != 0)
	{
		delete[] g_quadrantsEdges;
	}

	if (g_queryResults != 0)
	{
		delete[] g_queryResults;
	}
}
#endif