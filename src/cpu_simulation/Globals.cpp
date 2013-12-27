#include <Globals.h>
#include <Procedures.h>

#include <memory>

//////////////////////////////////////////////////////////////////////////
Configuration* g_configuration = 0;
//////////////////////////////////////////////////////////////////////////
RoadNetworkGraph::Graph* g_graph = 0;
//////////////////////////////////////////////////////////////////////////
unsigned char** g_workQueuesBuffers1 = 0;
//////////////////////////////////////////////////////////////////////////
unsigned char** g_workQueuesBuffers2 = 0;
//////////////////////////////////////////////////////////////////////////
StaticMarshallingQueue** g_workQueues1 = 0;
//////////////////////////////////////////////////////////////////////////
StaticMarshallingQueue** g_workQueues2 = 0;
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
void initializeWorkQueues()
{
	disposeWorkQueues();

	unsigned int capacity = g_configuration->maxWorkQueueCapacity;
	unsigned int itemSize = MathExtras::max(sizeof(Road), sizeof(Branch));
	unsigned int bufferSize = capacity * itemSize;

	g_workQueuesBuffers1 = new unsigned char*[NUM_PROCEDURES];
	g_workQueuesBuffers2 = new unsigned char*[NUM_PROCEDURES];

	g_workQueues1 = new StaticMarshallingQueue*[NUM_PROCEDURES];
	g_workQueues2 = new StaticMarshallingQueue*[NUM_PROCEDURES];
	for (unsigned int i = 0; i < NUM_PROCEDURES; i++)
	{
		g_workQueuesBuffers1[i] = new unsigned char[bufferSize];
		g_workQueuesBuffers2[i] = new unsigned char[bufferSize];
		g_workQueues1[i] = new StaticMarshallingQueue(g_workQueuesBuffers1[i], capacity, itemSize);
		g_workQueues2[i] = new StaticMarshallingQueue(g_workQueuesBuffers2[i], capacity, itemSize);
	}
}

//////////////////////////////////////////////////////////////////////////
void disposeWorkQueues()
{
	if (g_workQueues1 != 0)
	{
		for (unsigned int i = 0; i < NUM_PROCEDURES; i++)
		{
			delete g_workQueues1[i];
		}
		delete[] g_workQueues1;
		g_workQueues1 = 0;
	}

	if (g_workQueues2 != 0)
	{
		for (unsigned int i = 0; i < NUM_PROCEDURES; i++)
		{
			delete g_workQueues2[i];
		}
		delete[] g_workQueues2;
		g_workQueues2 = 0;
	}

	if (g_workQueuesBuffers1 != 0)
	{
		for (unsigned int i = 0; i < NUM_PROCEDURES; i++)
		{
			delete[] g_workQueuesBuffers1[i];
		}
		delete[] g_workQueuesBuffers1;
		g_workQueuesBuffers1 = 0;
	}

	if (g_workQueuesBuffers2 != 0)
	{
		for (unsigned int i = 0; i < NUM_PROCEDURES; i++)
		{
			delete[] g_workQueuesBuffers2[i];
		}
		delete[] g_workQueuesBuffers2;
		g_workQueuesBuffers2 = 0;
	}
}

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
		g_populationDensitiesSamplingBuffer = 0;
	}

	if (g_distancesSamplingBuffer != 0)
	{
		delete[] g_distancesSamplingBuffer;
		g_distancesSamplingBuffer = 0;
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
		g_vertices = 0;
	}

	if (g_edges != 0)
	{
		delete[] g_edges;
		g_edges = 0;
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