#include <RoadNetworkGraphGenerator.h>
#include <Constants.h>
#include <CpuGpuCompatibility.h>
#include <Procedures.h>
#include <Road.h>
#include <Branch.h>
#ifdef USE_QUADTREE
#include <Quadtree.h>
#include <Quadrant.h>
#include <QuadrantEdges.h>
#endif
#include <Primitive.h>
#include <BaseGraph.h>
#include <Graph.h>
#include <MathExtras.h>
#include <Box2D.h>
#include <MinimalCycleBasis.h>
#include <ConvexHull.h>
#include <OBB2D.h>
#include <VectorMath.h>
#include <Context.cuh>
#include <WorkQueue.cuh>

#include <exception>
#include <memory>

// DEBUG:
#include <iostream>

#define SAFE_MALLOC_ON_HOST(__variable, __type, __amount) \
	__variable = 0; \
	__variable = (__type*)malloc(sizeof(__type) * __amount); \
	if (__variable == 0) \
	{ \
		throw std::exception("#__variable: insufficient memory"); \
	}

#ifdef USE_CUDA
#include <cutil.h>
#include <cutil_timer.h>
#define NUM_BLOCKS 1
#define NUM_THREADS 128
#define SAFE_MALLOC_ON_DEVICE(__variable, __type, __amount) cudaCheckedCall(cudaMalloc((void**)&__variable, sizeof(__type) * __amount))
#define SAFE_FREE_ON_DEVICE(__variable) cudaCheckedCall(cudaFree(__variable))
#define MEMCPY_HOST_TO_DEVICE(__destination, __source, __size) cudaCheckedCall(cudaMemcpy(__destination, __source, __size, cudaMemcpyHostToDevice))
#define MEMCPY_DEVICE_TO_DEVICE(__destination, __source, __size) cudaCheckedCall(cudaMemcpy(__destination, __source, __size, cudaMemcpyDeviceToDevice))
#define MEMCPY_DEVICE_TO_HOST(__destination, __source, __size) cudaCheckedCall(cudaMemcpy(__destination, __source, __size, cudaMemcpyDeviceToHost))
#define MEMSET_ON_DEVICE(__variable, __value, __size) cudaCheckedCall(cudaMemset(__variable, __value, __size))
#define INVOKE_GLOBAL_CODE(__function, __numBlocks, __numThreads) \
	__function<<<__numBlocks, __numThreads>>>(); \
	cudaCheckError()
#define INVOKE_GLOBAL_CODE1(__function, __numBlocks, __numThreads, __arg1) \
	__function<<<__numBlocks, __numThreads>>>(__arg1); \
	cudaCheckError()
#define INVOKE_GLOBAL_CODE2(__function, __numBlocks, __numThreads, __arg1, __arg2) \
	__function<<<__numBlocks, __numThreads>>>(__arg1, __arg2); \
	cudaCheckError()
#define INVOKE_GLOBAL_CODE3(__function, __numBlocks, __numThreads, __arg1, __arg2, __arg3) \
	__function<<<__numBlocks, __numThreads>>>(__arg1, __arg2, __arg3); \
	cudaCheckError()
#define INVOKE_GLOBAL_CODE4(__function, __numBlocks, __numThreads, __arg1, __arg2, __arg3, __arg4) \
	__function<<<__numBlocks, __numThreads>>>(__arg1, __arg2, __arg3, __arg4); \
	cudaCheckError()
#define INVOKE_GLOBAL_CODE5(__function, __numBlocks, __numThreads, __arg1, __arg2, __arg3, __arg4, __arg5) \
	__function<<<__numBlocks, __numThreads>>>(__arg1, __arg2, __arg3, __arg4, __arg5); \
	cudaCheckError()
#define INVOKE_GLOBAL_CODE6(__function, __numBlocks, __numThreads, __arg1, __arg2, __arg3, __arg4, __arg5, __arg6) \
	__function<<<__numBlocks, __numThreads>>>(__arg1, __arg2, __arg3, __arg4, __arg5, __arg6); \
	cudaCheckError()
#define INVOKE_GLOBAL_CODE7(__function, __numBlocks, __numThreads, __arg1, __arg2, __arg3, __arg4, __arg5, __arg6, __arg7) \
	__function<<<__numBlocks, __numThreads>>>(__arg1, __arg2, __arg3, __arg4, __arg5, __arg6, __arg7); \
	cudaCheckError()
#define INVOKE_GLOBAL_CODE8(__function, __numBlocks, __numThreads, __arg1, __arg2, __arg3, __arg4, __arg5, __arg6, __arg7, __arg8) \
	__function<<<__numBlocks, __numThreads>>>(__arg1, __arg2, __arg3, __arg4, __arg5, __arg6, __arg7, __arg8); \
	cudaCheckError()
#define INVOKE_GLOBAL_CODE9(__function, __numBlocks, __numThreads, __arg1, __arg2, __arg3, __arg4, __arg5, __arg6, __arg7, __arg8, __arg9) \
	__function<<<__numBlocks, __numThreads>>>(__arg1, __arg2, __arg3, __arg4, __arg5, __arg6, __arg7, __arg8, __arg9); \
	cudaCheckError()
#define INVOKE_GLOBAL_CODE10(__function, __numBlocks, __numThreads, __arg1, __arg2, __arg3, __arg4, __arg5, __arg6, __arg7, __arg8, __arg9, __arg10) \
	__function<<<__numBlocks, __numThreads>>>(__arg1, __arg2, __arg3, __arg4, __arg5, __arg6, __arg7, __arg8, __arg9, __arg10); \
	cudaCheckError()
#define INVOKE_GLOBAL_CODE11(__function, __numBlocks, __numThreads, __arg1, __arg2, __arg3, __arg4, __arg5, __arg6, __arg7, __arg8, __arg9, __arg10, __arg11) \
	__function<<<__numBlocks, __numThreads>>>(__arg1, __arg2, __arg3, __arg4, __arg5, __arg6, __arg7, __arg8, __arg9, __arg10, __arg11); \
	cudaCheckError()
#define INVOKE_GLOBAL_CODE12(__function, __numBlocks, __numThreads, __arg1, __arg2, __arg3, __arg4, __arg5, __arg6, __arg7, __arg8, __arg9, __arg10, __arg11, __arg12) \
	__function<<<__numBlocks, __numThreads>>>(__arg1, __arg2, __arg3, __arg4, __arg5, __arg6, __arg7, __arg8, __arg9, __arg10, __arg11, __arg12); \
	cudaCheckError()
#define CREATE_TIMER(x) createTimer(x)
#define START_TIMER(x) startTimer(x)
#define STOP_TIMER(x) stopTimer(x)
#define GET_TIMER_ELAPSED_TIME(x, y) getTimerElapsedTime(x, y)
#define DESTROY_TIMER(x) destroyTimer(x)
#else
#include <Timer.h>
#define SAFE_MALLOC_ON_DEVICE(__variable, __type, __amount) \
	__variable = 0; \
	__variable = (__type*)malloc(sizeof(__type) * __amount); \
	if (__variable == 0) \
	{ \
		throw std::exception("#__variable: insufficient memory"); \
	}
#define SAFE_FREE_ON_DEVICE(__variable) free(__variable)
#define MEMCPY_HOST_TO_DEVICE(__destination, __source, __size) memcpy(__destination, __source, __size)
#define MEMCPY_DEVICE_TO_DEVICE(__destination, __source, __size) memcpy(__destination, __source, __size)
#define MEMCPY_DEVICE_TO_HOST(__destination, __source, __size) memcpy(__destination, __source, __size)
#define MEMSET_ON_DEVICE(__variable, __value, __size) memset(__variable, __value, __size)
#define INVOKE_GLOBAL_CODE(__function, __numBlocks, __numThreads) __function()
#define INVOKE_GLOBAL_CODE1(__function, __numBlocks, __numThreads, __arg1) __function(__arg1)
#define INVOKE_GLOBAL_CODE2(__function, __numBlocks, __numThreads, __arg1, __arg2) __function(__arg1, __arg2)
#define INVOKE_GLOBAL_CODE3(__function, __numBlocks, __numThreads, __arg1, __arg2, __arg3) __function(__arg1, __arg2, __arg3)
#define INVOKE_GLOBAL_CODE4(__function, __numBlocks, __numThreads, __arg1, __arg2, __arg3, __arg4) __function(__arg1, __arg2, __arg3, __arg4)
#define INVOKE_GLOBAL_CODE5(__function, __numBlocks, __numThreads, __arg1, __arg2, __arg3, __arg4, __arg5) __function(__arg1, __arg2, __arg3, __arg4, __arg5)
#define INVOKE_GLOBAL_CODE6(__function, __numBlocks, __numThreads, __arg1, __arg2, __arg3, __arg4, __arg5, __arg6) __function(__arg1, __arg2, __arg3, __arg4, __arg5, __arg6)
#define INVOKE_GLOBAL_CODE7(__function, __numBlocks, __numThreads, __arg1, __arg2, __arg3, __arg4, __arg5, __arg6, __arg7) __function(__arg1, __arg2, __arg3, __arg4, __arg5, __arg6, __arg7)
#define INVOKE_GLOBAL_CODE8(__function, __numBlocks, __numThreads, __arg1, __arg2, __arg3, __arg4, __arg5, __arg6, __arg7, __arg8) __function(__arg1, __arg2, __arg3, __arg4, __arg5, __arg6, __arg7, __arg8)
#define INVOKE_GLOBAL_CODE9(__function, __numBlocks, __numThreads, __arg1, __arg2, __arg3, __arg4, __arg5, __arg6, __arg7, __arg8, __arg9) __function(__arg1, __arg2, __arg3, __arg4, __arg5, __arg6, __arg7, __arg8, __arg9)
#define INVOKE_GLOBAL_CODE10(__function, __numBlocks, __numThreads, __arg1, __arg2, __arg3, __arg4, __arg5, __arg6, __arg7, __arg8, __arg9, __arg10) __function(__arg1, __arg2, __arg3, __arg4, __arg5, __arg6, __arg7, __arg8, __arg9, __arg10)
#define INVOKE_GLOBAL_CODE11(__function, __numBlocks, __numThreads, __arg1, __arg2, __arg3, __arg4, __arg5, __arg6, __arg7, __arg8, __arg9, __arg10, __arg11) __function(__arg1, __arg2, __arg3, __arg4, __arg5, __arg6, __arg7, __arg8, __arg9, __arg10, __arg11)
#define INVOKE_GLOBAL_CODE12(__function, __numBlocks, __numThreads, __arg1, __arg2, __arg3, __arg4, __arg5, __arg6, __arg7, __arg8, __arg9, __arg10, __arg11, __arg12) __function(__arg1, __arg2, __arg3, __arg4, __arg5, __arg6, __arg7, __arg8, __arg9, __arg10, __arg11, __arg12)
#define CREATE_TIMER(x) Timer x
#define START_TIMER(x) x.start()
#define STOP_TIMER(x) x.end()
#define GET_TIMER_ELAPSED_TIME(x, y) y = x.elapsedTime()
#define DESTROY_TIMER(x)
#endif

//////////////////////////////////////////////////////////////////////////
//	DEVICE VARIABLES
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE Graph* g_dGraph;
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE Configuration* g_dConfiguration;
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE ImageMap* g_dPopulationDensityMap;
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE ImageMap* g_dWaterBodiesMap;
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE ImageMap* g_dBlockadesMap;
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE ImageMap* g_dNaturalPatternMap;
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE ImageMap* g_dRadialPatternMap;
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE ImageMap* g_dRasterPatternMap;
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE Context* g_dContext;
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE WorkQueue* g_dWorkQueues1;
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE WorkQueue* g_dWorkQueues2;
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE Vertex* g_dVertices;
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE Edge* g_dEdges;
#ifdef USE_QUADTREE
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE QuadTree* g_dQuadtree;
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE Quadrant* g_dQuadrants;
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE QuadrantEdges* g_dQuadrantsEdges;
#endif
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE unsigned char* g_dPopulationDensityMapData;
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE unsigned char* g_dWaterBodiesMapData;
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE unsigned char* g_dBlockadesMapData;
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE unsigned char* g_dNaturalPatternMapData;
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE unsigned char* g_dRadialPatternMapData;
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE unsigned char* g_dRasterPatternMapData;
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE Primitive* g_dPrimitives;

//////////////////////////////////////////////////////////////////////////
GLOBAL_CODE void initializeImageMapOnDevice(ImageMap* imageMap, unsigned int width, unsigned int height, unsigned char* data)
{
	imageMap->setWidth(width);
	imageMap->setHeight(height);
	imageMap->setData(data);
}

//////////////////////////////////////////////////////////////////////////
GLOBAL_CODE void initializeContext(Context* context,
								   Graph* graph,
								   Configuration* configuration,
								   ImageMap* populationDensityMap,
								   ImageMap* waterBodiesMap,
								   ImageMap* blockadesMap,
								   ImageMap* naturalPatternMap,
								   ImageMap* radialPatternMap,
								   ImageMap* rasterPatternMap,
								   Primitive* primitives)
{
	context->graph = graph;
	context->configuration = configuration;
	context->populationDensityMap = populationDensityMap;
	context->waterBodiesMap = waterBodiesMap;
	context->blockadesMap = blockadesMap;
	context->naturalPatternMap = naturalPatternMap;
	context->radialPatternMap = radialPatternMap;
	context->rasterPatternMap = rasterPatternMap;
	context->primitives = primitives;
}

//////////////////////////////////////////////////////////////////////////
#define allocateAndInitializeImageMap(__name1, __name2) \
	if (__name1##Map.hasData()) \
	{ \
		unsigned int mapSize = __name1##Map.getWidth() * __name1##Map.getHeight(); \
		SAFE_MALLOC_ON_DEVICE(g_d##__name2##MapData, unsigned char, mapSize); \
		MEMCPY_HOST_TO_DEVICE(g_d##__name2##MapData, __name1##Map.getData(), sizeof(unsigned char) * mapSize); \
		SAFE_MALLOC_ON_DEVICE(g_d##__name2##Map, ImageMap, 1); \
		INVOKE_GLOBAL_CODE4(initializeImageMapOnDevice, 1, 1, g_d##__name2##Map, __name1##Map.getWidth(), __name1##Map.getHeight(), g_d##__name2##MapData); \
	}

//////////////////////////////////////////////////////////////////////////
void RoadNetworkGraphGenerator::notifyObservers(Graph* graph, unsigned int numPrimitives, Primitive* primitives)
{
	for (unsigned int i = 0; i < observers.size(); i++)
	{
		observers[i]->update(graph, numPrimitives, primitives);
	}
}

//////////////////////////////////////////////////////////////////////////
void RoadNetworkGraphGenerator::copyGraphToDevice(Graph* graph)
{
#ifdef USE_QUADTREE
	MEMCPY_HOST_TO_DEVICE(g_dQuadrants, graph->quadtree->quadrants, sizeof(Quadrant) * configuration.maxQuadrants);
	MEMCPY_HOST_TO_DEVICE(g_dQuadrantsEdges, graph->quadtree->quadrantsEdges, sizeof(QuadrantEdges) * configuration.maxQuadrants);
#ifdef COLLECT_STATISTICS
	INVOKE_GLOBAL_CODE10(updateNonPointerFields, 1, 1, g_dQuadtree, (int)graph->quadtree->numQuadrantEdges, graph->quadtree->worldBounds, graph->quadtree->maxDepth, graph->quadtree->maxQuadrants, graph->quadtree->totalNumQuadrants, graph->quadtree->numLeafQuadrants, (unsigned long)graph->quadtree->numCollisionChecks, (unsigned int)graph->quadtree->maxEdgesPerQuadrantInUse, (unsigned int)graph->quadtree->maxResultsPerQueryInUse);
#else
	INVOKE_GLOBAL_CODE7(updateNonPointerFields, 1, 1, g_dQuadtree, (int)graph->quadtree->numQuadrantEdges, graph->quadtree->worldBounds, graph->quadtree->maxDepth, graph->quadtree->maxQuadrants, graph->quadtree->totalNumQuadrants, graph->quadtree->numLeafQuadrants);
#endif
#endif

	MEMCPY_HOST_TO_DEVICE(g_dVertices, graph->vertices, sizeof(Vertex) * configuration.maxVertices);
	MEMCPY_HOST_TO_DEVICE(g_dEdges, graph->edges, sizeof(Edge) * configuration.maxEdges);
#ifdef COLLECT_STATISTICS
	INVOKE_GLOBAL_CODE6(updateNonPointerFields, 1, 1, g_dGraph, (int)graph->numVertices, (int)graph->numEdges, graph->maxVertices, graph->maxEdges, (unsigned long)graph->numCollisionChecks);
#else
	INVOKE_GLOBAL_CODE5(updateNonPointerFields, 1, 1, g_dGraph, (int)graph->numVertices, (int)graph->numEdges, graph->maxVertices, graph->maxEdges);
#endif
}

//////////////////////////////////////////////////////////////////////////
void RoadNetworkGraphGenerator::copyGraphToHost(Graph* graph)
{
#ifdef USE_QUADTREE
	MEMCPY_DEVICE_TO_HOST(graph->quadtree->quadrants, g_dQuadrants, sizeof(Quadrant) * configuration.maxQuadrants);
	MEMCPY_DEVICE_TO_HOST(graph->quadtree->quadrantsEdges, g_dQuadrantsEdges, sizeof(QuadrantEdges) * configuration.maxQuadrants);
	Quadrant* quadrants = graph->quadtree->quadrants;
	QuadrantEdges* quadrantsEdges = graph->quadtree->quadrantsEdges;
	MEMCPY_DEVICE_TO_HOST(graph->quadtree, g_dQuadtree, sizeof(QuadTree));
	graph->quadtree->quadrants = quadrants;
	graph->quadtree->quadrantsEdges = quadrantsEdges;
#endif
	MEMCPY_DEVICE_TO_HOST(graph->vertices, g_dVertices, sizeof(Vertex) * configuration.maxVertices);
	MEMCPY_DEVICE_TO_HOST(graph->edges, g_dEdges, sizeof(Edge) * configuration.maxEdges);
#ifdef USE_QUADTREE
	QuadTree* quadtree = graph->quadtree;
#endif
	Vertex* vertices = graph->vertices;
	Edge* edges = graph->edges;
	MEMCPY_DEVICE_TO_HOST(graph, g_dGraph, sizeof(Graph));
#ifdef USE_QUADTREE
	graph->quadtree = quadtree;
#endif
	graph->vertices = vertices;
	graph->edges = edges;
}

//////////////////////////////////////////////////////////////////////////
void RoadNetworkGraphGenerator::execute()
{
	CREATE_TIMER(PrimaryRoadNetworkExpansion);
	CREATE_TIMER(SecondaryRoadNetworkExpansion);

	allocateAndInitializeImageMap(populationDensity, PopulationDensity);
	allocateAndInitializeImageMap(waterBodies, WaterBodies);
	allocateAndInitializeImageMap(blockades, Blockades);
	allocateAndInitializeImageMap(naturalPattern, NaturalPattern);
	allocateAndInitializeImageMap(radialPattern, RadialPattern);
	allocateAndInitializeImageMap(rasterPattern, RasterPattern);

#ifdef USE_QUADTREE
	QuadTree* quadtree;
	Quadrant* quadrants;
	QuadrantEdges* quadrantsEdges;

	SAFE_MALLOC_ON_HOST(quadtree, QuadTree, 1);
	SAFE_MALLOC_ON_HOST(quadrants, Quadrant, configuration.maxQuadrants);
	SAFE_MALLOC_ON_HOST(quadrantsEdges, QuadrantEdges, configuration.maxQuadrants);

	memset(quadrants, 0, sizeof(Quadrant) * configuration.maxQuadrants);
	memset(quadrantsEdges, 0, sizeof(QuadrantEdges) * configuration.maxQuadrants);

	SAFE_MALLOC_ON_DEVICE(g_dQuadtree, QuadTree, 1);
	SAFE_MALLOC_ON_DEVICE(g_dQuadrants, Quadrant, configuration.maxQuadrants);
	SAFE_MALLOC_ON_DEVICE(g_dQuadrantsEdges, QuadrantEdges, configuration.maxQuadrants);

	MEMSET_ON_DEVICE(g_dQuadrants, 0, sizeof(Quadrant) * configuration.maxQuadrants);
	MEMSET_ON_DEVICE(g_dQuadrantsEdges, 0, sizeof(QuadrantEdges) * configuration.maxQuadrants);

	Box2D worldBounds(0.0f, 0.0f, (float)configuration.worldWidth, (float)configuration.worldHeight);

	initializeQuadtreeOnHost(quadtree, worldBounds, configuration.quadtreeDepth, configuration.maxQuadrants, quadrants, quadrantsEdges);
	INVOKE_GLOBAL_CODE6(initializeQuadtreeOnDevice, 1, 1, g_dQuadtree, worldBounds, configuration.quadtreeDepth, configuration.maxQuadrants, g_dQuadrants, g_dQuadrantsEdges);
#endif

	Graph* graph;
	Vertex* vertices;
	Edge* edges;

	SAFE_MALLOC_ON_HOST(graph, Graph, 1);
	SAFE_MALLOC_ON_HOST(vertices, Vertex, configuration.maxVertices);
	SAFE_MALLOC_ON_HOST(edges, Edge, configuration.maxEdges);

	memset(vertices, 0, sizeof(Vertex) * configuration.maxVertices);
	memset(edges, 0, sizeof(Edge) * configuration.maxEdges);

	SAFE_MALLOC_ON_DEVICE(g_dGraph, Graph, 1);
	SAFE_MALLOC_ON_DEVICE(g_dVertices, Vertex, configuration.maxVertices);
	SAFE_MALLOC_ON_DEVICE(g_dEdges, Edge, configuration.maxEdges);

	MEMSET_ON_DEVICE(g_dVertices, 0, sizeof(Vertex) * configuration.maxVertices);
	MEMSET_ON_DEVICE(g_dEdges, 0, sizeof(Edge) * configuration.maxEdges);
	
#ifdef USE_QUADTREE
	initializeGraphOnHost(graph, configuration.snapRadius, configuration.maxVertices, configuration.maxEdges, vertices, edges, quadtree);
	INVOKE_GLOBAL_CODE7(initializeGraphOnDevice, 1, 1, g_dGraph, configuration.snapRadius, configuration.maxVertices, configuration.maxEdges, g_dVertices, g_dEdges, g_dQuadtree);
#else
	initializeGraphOnHost(graph, configuration.snapRadius, configuration.maxVertices, configuration.maxEdges, vertices, edges);
	INVOKE_GLOBAL_CODE6(initializeGraphOnDevice, 1, 1, g_dGraph, configuration.snapRadius, configuration.maxVertices, configuration.maxEdges, g_dVertices, g_dEdges);
#endif

	SAFE_MALLOC_ON_DEVICE(g_dWorkQueues1, WorkQueue, NUM_PROCEDURES);
	SAFE_MALLOC_ON_DEVICE(g_dWorkQueues2, WorkQueue, NUM_PROCEDURES);

	WorkQueue* workQueues1;
	WorkQueue* workQueues2;

	SAFE_MALLOC_ON_HOST(workQueues1, WorkQueue, NUM_PROCEDURES);
	SAFE_MALLOC_ON_HOST(workQueues2, WorkQueue, NUM_PROCEDURES);

	memset(workQueues1, 0, sizeof(WorkQueue) * NUM_PROCEDURES);
	memset(workQueues2, 0, sizeof(WorkQueue) * NUM_PROCEDURES);

	// set highway spawn points
	for (unsigned int i = 0; i < configuration.numSpawnPoints; i++)
	{
		vml_vec2 spawnPoint = configuration.spawnPoints[i];
		int source = createVertex(graph, spawnPoint);
		workQueues1[EVALUATE_HIGHWAY].unsafePush(Highway(0, RoadAttributes(source, configuration.highwayLength, 0), UNASSIGNED));
		workQueues1[EVALUATE_HIGHWAY].unsafePush(Highway(0, RoadAttributes(source, configuration.highwayLength, -HALF_PI), UNASSIGNED));
		workQueues1[EVALUATE_HIGHWAY].unsafePush(Highway(0, RoadAttributes(source, configuration.highwayLength, HALF_PI), UNASSIGNED));
		workQueues1[EVALUATE_HIGHWAY].unsafePush(Highway(0, RoadAttributes(source, configuration.highwayLength, PI), UNASSIGNED));
	}
	copyGraphToDevice(graph);

	SAFE_MALLOC_ON_DEVICE(g_dPrimitives, Primitive, configuration.maxPrimitives);

	MEMCPY_HOST_TO_DEVICE(g_dWorkQueues1, workQueues1, sizeof(WorkQueue) * NUM_PROCEDURES);
	MEMCPY_HOST_TO_DEVICE(g_dWorkQueues2, workQueues2, sizeof(WorkQueue) * NUM_PROCEDURES);

	SAFE_MALLOC_ON_DEVICE(g_dConfiguration, Configuration, 1);
	MEMCPY_HOST_TO_DEVICE(g_dConfiguration, const_cast<Configuration*>(&configuration), sizeof(Configuration));

	SAFE_MALLOC_ON_DEVICE(g_dContext, Context, 1);
	INVOKE_GLOBAL_CODE10(initializeContext, 1, 1, 
		g_dContext,
		g_dGraph, 
		g_dConfiguration, 
		g_dPopulationDensityMap,
		g_dWaterBodiesMap,
		g_dBlockadesMap,
		g_dNaturalPatternMap,
		g_dRadialPatternMap,
		g_dRasterPatternMap,
		g_dPrimitives);

	START_TIMER(PrimaryRoadNetworkExpansion);

	expand(configuration.maxHighwayDerivation);

	STOP_TIMER(PrimaryRoadNetworkExpansion);

	float elapsedTime;
	GET_TIMER_ELAPSED_TIME(PrimaryRoadNetworkExpansion, elapsedTime);
	std::cout << "Primary Road Network Expansion: " << elapsedTime << " (ms)" << std::endl;

	copyGraphToHost(graph);

	BaseGraph* graphCopy;
	Vertex* verticesCopy;
	Edge* edgesCopy;

	SAFE_MALLOC_ON_HOST(graphCopy, BaseGraph, 1);
	SAFE_MALLOC_ON_HOST(verticesCopy, Vertex, configuration.maxVertices);
	SAFE_MALLOC_ON_HOST(edgesCopy, Edge, configuration.maxEdges);

	graphCopy->vertices = verticesCopy;
	graphCopy->edges = edgesCopy;

	memcpy(graphCopy->vertices, graph->vertices, sizeof(Vertex) * configuration.maxVertices);
	memcpy(graphCopy->edges, graph->edges, sizeof(Edge) * configuration.maxEdges);
	
	graphCopy->numVertices = graph->numVertices;
	graphCopy->numEdges = graph->numEdges;
	Primitive* primitives;
	
	SAFE_MALLOC_ON_HOST(primitives, Primitive, configuration.maxPrimitives);

	memset(primitives, 0, sizeof(Primitive) * configuration.maxPrimitives);

	// extract the allotments from graph copy
	allocateExtractionBuffers(configuration.maxVertices, configuration.maxEdgeSequences, configuration.maxVisitedVertices);
	unsigned int numPrimitives = extractPrimitives(graphCopy, primitives, configuration.maxPrimitives);
	freeExtractionBuffers();

	free(graphCopy);
	free(verticesCopy);
	free(edgesCopy);

	MEMCPY_HOST_TO_DEVICE(g_dPrimitives, primitives, sizeof(Primitive) * configuration.maxPrimitives);

	for (unsigned int i = 0; i < NUM_PROCEDURES; i++)
	{
		workQueues1[i].clear();
		workQueues2[i].clear();
	}

	maxPrimitiveSize = 0;
	// set street spawn points
	for (unsigned int i = 0; i < numPrimitives; i++)
	{
		Primitive& primitive = primitives[i];

		if (primitive.type != MINIMAL_CYCLE)
		{
			continue;
		}

		maxPrimitiveSize = MathExtras::max(maxPrimitiveSize, primitive.numEdges);

		vml_vec2 centroid;
		float area;
		MathExtras::getPolygonInfo(primitive.vertices, primitive.numVertices, area, centroid);
		if (area < configuration.minBlockArea)
		{
			continue;
		}

		float angle;
		ConvexHull convexHull(primitive.vertices, primitive.numVertices);
		OBB2D obb(convexHull.hullPoints, convexHull.numHullPoints);
		angle = vml_angle(obb.axis[1], vml_vec2(0.0f, 1.0f));

		int source = createVertex(graph, centroid);
		workQueues1[EVALUATE_STREET].unsafePush(Street(0, RoadAttributes(source, configuration.streetLength, angle), StreetRuleAttributes(0, i), UNASSIGNED));
		workQueues1[EVALUATE_STREET].unsafePush(Street(0, RoadAttributes(source, configuration.streetLength, -HALF_PI + angle), StreetRuleAttributes(0, i), UNASSIGNED));
		workQueues1[EVALUATE_STREET].unsafePush(Street(0, RoadAttributes(source, configuration.streetLength, HALF_PI + angle), StreetRuleAttributes(0, i), UNASSIGNED));
		workQueues1[EVALUATE_STREET].unsafePush(Street(0, RoadAttributes(source, configuration.streetLength, PI + angle), StreetRuleAttributes(0, i), UNASSIGNED));
	}
	copyGraphToDevice(graph);

	MEMCPY_HOST_TO_DEVICE(g_dWorkQueues1, workQueues1, sizeof(WorkQueue) * NUM_PROCEDURES);
	MEMCPY_HOST_TO_DEVICE(g_dWorkQueues2, workQueues2, sizeof(WorkQueue) * NUM_PROCEDURES);

	START_TIMER(SecondaryRoadNetworkExpansion);

	expand(configuration.maxStreetDerivation);

	STOP_TIMER(SecondaryRoadNetworkExpansion);

	GET_TIMER_ELAPSED_TIME(SecondaryRoadNetworkExpansion, elapsedTime);
	std::cout << "Secondary Road Network Expansion: " << elapsedTime << " (ms)" << std::endl;

	copyGraphToHost(graph);

	MEMCPY_DEVICE_TO_HOST(primitives, g_dPrimitives, sizeof(Primitive) * configuration.maxPrimitives);

	notifyObservers(graph, numPrimitives, primitives);

	SAFE_FREE_ON_DEVICE(g_dContext);
	SAFE_FREE_ON_DEVICE(g_dConfiguration);
	SAFE_FREE_ON_DEVICE(g_dWorkQueues1);
	SAFE_FREE_ON_DEVICE(g_dWorkQueues2);
	SAFE_FREE_ON_DEVICE(g_dPopulationDensityMap);
	SAFE_FREE_ON_DEVICE(g_dWaterBodiesMap);
	SAFE_FREE_ON_DEVICE(g_dBlockadesMap);
	SAFE_FREE_ON_DEVICE(g_dNaturalPatternMap);
	SAFE_FREE_ON_DEVICE(g_dRadialPatternMap);
	SAFE_FREE_ON_DEVICE(g_dRasterPatternMap);
	SAFE_FREE_ON_DEVICE(g_dPopulationDensityMapData);
	SAFE_FREE_ON_DEVICE(g_dWaterBodiesMapData);
	SAFE_FREE_ON_DEVICE(g_dBlockadesMapData);
	SAFE_FREE_ON_DEVICE(g_dNaturalPatternMapData);
	SAFE_FREE_ON_DEVICE(g_dRadialPatternMapData);
	SAFE_FREE_ON_DEVICE(g_dRasterPatternMapData);
	SAFE_FREE_ON_DEVICE(g_dPrimitives);

#ifdef USE_QUADTREE
	SAFE_FREE_ON_DEVICE(g_dQuadrants);
	SAFE_FREE_ON_DEVICE(g_dQuadrantsEdges);
	SAFE_FREE_ON_DEVICE(g_dQuadtree);
#endif
	SAFE_FREE_ON_DEVICE(g_dVertices);
	SAFE_FREE_ON_DEVICE(g_dEdges);
	SAFE_FREE_ON_DEVICE(g_dGraph);

	free(primitives);
	free(workQueues1);
	free(workQueues2);
#ifdef USE_QUADTREE
	free(quadrants);
	free(quadrantsEdges);
	free(quadtree);
#endif
	free(graph);
	free(vertices);
	free(edges);

	DESTROY_TIMER(PrimaryRoadNetworkExpansion);
	DESTROY_TIMER(SecondaryRoadNetworkExpansion);
}

#ifdef USE_CUDA
//////////////////////////////////////////////////////////////////////////
__device__ volatile int g_dCounterLevel1;
//////////////////////////////////////////////////////////////////////////
__device__ volatile int g_dCounterLevel2;

//////////////////////////////////////////////////////////////////////////
__global__ void initializeKernel()
{
	g_dCounterLevel1 = 0;
	g_dCounterLevel2 = 0;
}

//////////////////////////////////////////////////////////////////////////
__global__ void gpuKernel(unsigned int numDerivations, WorkQueue* workQueues1, WorkQueue* workQueues2, Context* context);

//////////////////////////////////////////////////////////////////////////
void RoadNetworkGraphGenerator::expand(unsigned int numDerivations)
{
	initializeKernel<<<1, 1>>>();
	cudaCheckError();
	gpuKernel<<<NUM_BLOCKS, NUM_THREADS>>>(numDerivations, g_dWorkQueues1, g_dWorkQueues2, g_dContext);
	cudaDeviceSynchronize();
	cudaCheckError();
}

//////////////////////////////////////////////////////////////////////////
__global__ void gpuKernel(unsigned int numDerivations, WorkQueue* workQueues1, WorkQueue* workQueues2, Context* context)
{
	__shared__ WorkQueue* frontQueues;
	__shared__ WorkQueue* backQueues;
	__shared__ volatile unsigned int derivation;
	__shared__ unsigned int reservedPops;
	__shared__ unsigned int head;
	__shared__ volatile unsigned int run;
	__shared__ unsigned int currentQueue;

	if (threadIdx.x == 0)
	{
		derivation = 0;
		frontQueues = workQueues1;
		backQueues = workQueues2;
	}

	__syncthreads();

	while (derivation < numDerivations)
	{
		if (threadIdx.x == 0)
		{
			atomicAdd((int*)&g_dCounterLevel1, 1);
			atomicAdd((int*)&g_dCounterLevel2, 1);
			run = 1;
			currentQueue = (blockIdx.x % NUM_PROCEDURES);
		}
		
		__syncthreads();

		while (run > 0)
		{
			// block optimization
			if (threadIdx.x == 0)
			{
				frontQueues[currentQueue].reservePops(blockDim.x, &head, &reservedPops);

				unsigned int queueShifts = 0;
				// round-robin through all the queues until pops can be reserved
				while (reservedPops == 0 && ++queueShifts < NUM_PROCEDURES)
				{
					currentQueue = (currentQueue + 1) % NUM_PROCEDURES;
					frontQueues[currentQueue].reservePops(blockDim.x, &head, &reservedPops);
				}
			}

			__syncthreads();

			if (threadIdx.x < reservedPops)
			{
				switch (currentQueue)
				{
				case EVALUATE_HIGHWAY_BRANCH:
					{
						HighwayBranch highwayBranch;
						frontQueues[EVALUATE_HIGHWAY_BRANCH].popReserved(head + threadIdx.x, highwayBranch);
						EvaluateHighwayBranch::execute(highwayBranch, context, backQueues);
					}
					break;
				case EVALUATE_HIGHWAY:
					{
						Highway highway;
						frontQueues[EVALUATE_HIGHWAY].popReserved(head + threadIdx.x, highway);
						EvaluateHighway::execute(highway, context, backQueues);
					}
					break;
				case INSTANTIATE_HIGHWAY:
					{
						Highway highway;
						frontQueues[INSTANTIATE_HIGHWAY].popReserved(head + threadIdx.x, highway);
						InstantiateHighway::execute(highway, context, backQueues);
					}
					break;
				case EVALUATE_STREET_BRANCH:
					{
						StreetBranch streetBranch;
						frontQueues[EVALUATE_STREET_BRANCH].popReserved(head + threadIdx.x, streetBranch);
						EvaluateStreetBranch::execute(streetBranch, context, backQueues);
					}
					break;
				case EVALUATE_STREET:
					{
						Street street;
						frontQueues[EVALUATE_STREET].popReserved(head + threadIdx.x, street);
						EvaluateStreet::execute(street, context, backQueues);
					}
					break;
				case INSTANTIATE_STREET:
					{
						Street street;
						frontQueues[INSTANTIATE_STREET].popReserved(head + threadIdx.x, street);
						InstantiateStreet::execute(street, context, backQueues);
					}
					break;
				default:
					THROW_EXCEPTION("invalid queue index");
				}
			}

			if (threadIdx.x == 0)
			{
				if (reservedPops == 0 && run == 1)
				{
					atomicSub((int*)&g_dCounterLevel1, 1), run = 2;
				}
				else if (reservedPops == 0 && run == 2 && g_dCounterLevel1 == 0)
				{
					atomicSub((int*)&g_dCounterLevel2, 1), run = 3;
				}
				else if (reservedPops == 0 && run == 3 && g_dCounterLevel2 == 0)
				{
					run = 0;
				}
				else if (reservedPops != 0 && run != 1)
				{
					if (run == 2)
					{
						atomicAdd((int*)&g_dCounterLevel1, 1), run = 1;
					}

					if (run == 3)
					{
						atomicAdd((int*)&g_dCounterLevel1, 1), atomicAdd((int*)&g_dCounterLevel2, 1), run = 1;
					}
				}
			}

			__syncthreads();
		}

		if (threadIdx.x == 0)
		{
			derivation++;
			WorkQueue* tmp = frontQueues;
			frontQueues = backQueues;
			backQueues = tmp;
		}

		__syncthreads();
	}
}
#else
//////////////////////////////////////////////////////////////////////////
void cpuKernel(unsigned int numDerivations, WorkQueue* workQueues1, WorkQueue* workQueues2, Context* context);

//////////////////////////////////////////////////////////////////////////
void RoadNetworkGraphGenerator::expand(unsigned int numDerivations)
{
	cpuKernel(numDerivations, g_dWorkQueues1, g_dWorkQueues2, g_dContext);
}

//////////////////////////////////////////////////////////////////////////
void cpuKernel(unsigned int numDerivations, WorkQueue* workQueues1, WorkQueue* workQueues2, Context* context)
{
	unsigned int derivations = 0;
	WorkQueue* frontQueues = workQueues1;
	WorkQueue* backQueues = workQueues2;

	while (derivations < numDerivations)
	{
		for (unsigned int currentQueue = 0; currentQueue < NUM_PROCEDURES; currentQueue++)
		{
			switch (currentQueue)
			{
			case EVALUATE_HIGHWAY_BRANCH:
				{
					HighwayBranch highwayBranch;
					while (frontQueues[EVALUATE_HIGHWAY_BRANCH].count > 0)
					{
						frontQueues[EVALUATE_HIGHWAY_BRANCH].unsafePop(highwayBranch);
						EvaluateHighwayBranch::execute(highwayBranch, context, backQueues);
					}
				}
				break;
			case EVALUATE_HIGHWAY:
				{
					Highway highway;
					while (frontQueues[EVALUATE_HIGHWAY].count > 0)
					{
						frontQueues[EVALUATE_HIGHWAY].unsafePop(highway);
						EvaluateHighway::execute(highway, context, backQueues);
					}
				}
				break;
			case INSTANTIATE_HIGHWAY:
				{
					Highway highway;
					while (frontQueues[INSTANTIATE_HIGHWAY].count > 0)
					{
						frontQueues[INSTANTIATE_HIGHWAY].unsafePop(highway);
						InstantiateHighway::execute(highway, context, backQueues);
					}
				}
				break;
			case EVALUATE_STREET_BRANCH:
				{
					StreetBranch streetBranch;
					while (frontQueues[EVALUATE_STREET_BRANCH].count > 0)
					{
						frontQueues[EVALUATE_STREET_BRANCH].unsafePop(streetBranch);
						EvaluateStreetBranch::execute(streetBranch, context, backQueues);
					}
				}
				break;
			case EVALUATE_STREET:
				{
					Street street;
					while (frontQueues[EVALUATE_STREET].count > 0)
					{
						frontQueues[EVALUATE_STREET].unsafePop(street);
						EvaluateStreet::execute(street, context, backQueues);
					}
				}
				break;
			case INSTANTIATE_STREET:
				{
					Street street;
					while (frontQueues[INSTANTIATE_STREET].count > 0)
					{
						frontQueues[INSTANTIATE_STREET].unsafePop(street);
						InstantiateStreet::execute(street, context, backQueues);
					}
				}
				break;
			default:
				THROW_EXCEPTION("invalid queue index");
			}
		}
		WorkQueue* tmp = frontQueues;
		frontQueues = backQueues;
		backQueues = tmp;
		derivations++;
	}
}
#endif