#include <RoadNetworkGraphGenerator.h>
#include <Constants.h>
#include <CpuGpuCompatibility.h>
#include <Procedures.h>
#include <Road.h>
#include <Branch.h>
#include <Quadtree.h>
#include <Quadrant.h>
#include <QuadrantEdges.h>
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
#include <Timer.h>
#include <GlobalVariables.cuh>

#include <exception>
#include <memory>

// DEBUG:
#include <iostream>

#define SAFE_MALLOC_ON_HOST(__variable, __type, __amount) \
	__variable = 0; \
	__variable = (__type*)malloc(sizeof(__type) * __amount); \
	if (__variable == 0) \
	{ \
		throw std::exception(#__variable": insufficient memory"); \
	}

#define CREATE_CPU_TIMER(x) Timer x
#define START_CPU_TIMER(x) x.start()
#define STOP_CPU_TIMER(x) x.end()
#define GET_CPU_TIMER_ELAPSED_TIME(x, y) y = x.elapsedTime()
#define DESTROY_CPU_TIMER(x)

#ifdef USE_CUDA
#include <cutil.h>
#include <cutil_timer.h>
#define NUM_EXPANSION_KERNEL_BLOCKS 12
#define NUM_EXPANSION_KERNEL_THREADS 256
#define NUM_COALESCE_KERNEL_BLOCKS_PER_QUADRANT 8
#define NUM_COALESCE_KERNEL_THREADS 256
#define SAFE_MALLOC_ON_DEVICE(__variable, __type, __amount) cudaCheckedCall(cudaMalloc((void**)&__variable, sizeof(__type) * __amount))
#define SAFE_FREE_ON_DEVICE(__variable) cudaCheckedCall(cudaFree(__variable))
#define MEMCPY_HOST_TO_DEVICE(__destination, __source, __size) cudaCheckedCall(cudaMemcpy(__destination, __source, __size, cudaMemcpyHostToDevice))
#define MEMCPY_DEVICE_TO_DEVICE(__destination, __source, __size) cudaCheckedCall(cudaMemcpy(__destination, __source, __size, cudaMemcpyDeviceToDevice))
#define MEMCPY_DEVICE_TO_HOST(__destination, __source, __size) cudaCheckedCall(cudaMemcpy(__destination, __source, __size, cudaMemcpyDeviceToHost))
#define MEMSET_ON_DEVICE(__variable, __value, __size) cudaCheckedCall(cudaMemset(__variable, __value, __size))
#define BIND_AS_TEXTURE2D(__deviceVariable, __texture, __width, __height) \
	{ \
		cudaChannelFormatDesc descriptor = cudaCreateChannelDesc<unsigned char>(); \
		cudaCheckedCall(cudaBindTexture2D(0, __texture, __deviceVariable, descriptor, __width, __height, sizeof(unsigned char) * __width)); \
	}
#define UNBIND_TEXTURE2D(__texture) cudaCheckedCall(cudaUnbindTexture(__texture))
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
#define CREATE_GPU_TIMER(x) createTimer(x)
#define START_GPU_TIMER(x) startTimer(x)
#define STOP_GPU_TIMER(x) stopTimer(x)
#define GET_GPU_TIMER_ELAPSED_TIME(x, y) getTimerElapsedTime(x, y)
#define DESTROY_GPU_TIMER(x) destroyTimer(x)
#else
#define SAFE_MALLOC_ON_DEVICE(__variable, __type, __amount) \
	__variable = 0; \
	__variable = (__type*)malloc(sizeof(__type) * __amount); \
	if (__variable == 0) \
	{ \
		throw std::exception(#__variable": insufficient memory"); \
	}
#define SAFE_FREE_ON_DEVICE(__variable) free(__variable)
#define MEMCPY_HOST_TO_DEVICE(__destination, __source, __size) memcpy(__destination, __source, __size)
#define MEMCPY_DEVICE_TO_DEVICE(__destination, __source, __size) memcpy(__destination, __source, __size)
#define MEMCPY_DEVICE_TO_HOST(__destination, __source, __size) memcpy(__destination, __source, __size)
#define MEMSET_ON_DEVICE(__variable, __value, __size) memset(__variable, __value, __size)
#define BIND_AS_TEXTURE2D(__deviceVariable, __texture, __width, __height) \
	__texture.width = __width; \
	__texture.height = __height; \
	__texture.data = __deviceVariable
#define UNBIND_TEXTURE2D(__texture)
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
#define CREATE_GPU_TIMER(x) CREATE_CPU_TIMER(x)
#define START_GPU_TIMER(x) START_CPU_TIMER(x)
#define STOP_GPU_TIMER(x) STOP_CPU_TIMER(x)
#define GET_GPU_TIMER_ELAPSED_TIME(x, y) GET_CPU_TIMER_ELAPSED_TIME(x, y)
#define DESTROY_GPU_TIMER(x) DESTROY_CPU_TIMER(x)
#endif

//////////////////////////////////////////////////////////////////////////
//	DEVICE VARIABLES
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
DEVICE_VARIABLE Graph* g_dGraph;
//////////////////////////////////////////////////////////////////////////
DEVICE_VARIABLE Configuration* g_dConfiguration;
//////////////////////////////////////////////////////////////////////////
DEVICE_VARIABLE ImageMap* g_dPopulationDensityMap;
//////////////////////////////////////////////////////////////////////////
DEVICE_VARIABLE ImageMap* g_dWaterBodiesMap;
//////////////////////////////////////////////////////////////////////////
DEVICE_VARIABLE ImageMap* g_dBlockadesMap;
//////////////////////////////////////////////////////////////////////////
DEVICE_VARIABLE ImageMap* g_dNaturalPatternMap;
//////////////////////////////////////////////////////////////////////////
DEVICE_VARIABLE ImageMap* g_dRadialPatternMap;
//////////////////////////////////////////////////////////////////////////
DEVICE_VARIABLE ImageMap* g_dRasterPatternMap;
//////////////////////////////////////////////////////////////////////////
DEVICE_VARIABLE Context* g_dContext;
//////////////////////////////////////////////////////////////////////////
DEVICE_VARIABLE WorkQueue* g_dWorkQueues1;
//////////////////////////////////////////////////////////////////////////
DEVICE_VARIABLE WorkQueue* g_dWorkQueues2;
//////////////////////////////////////////////////////////////////////////
DEVICE_VARIABLE Vertex* g_dVertices;
//////////////////////////////////////////////////////////////////////////
DEVICE_VARIABLE Edge* g_dEdges;
//////////////////////////////////////////////////////////////////////////
DEVICE_VARIABLE QuadTree* g_dQuadtree;
//////////////////////////////////////////////////////////////////////////
DEVICE_VARIABLE Quadrant* g_dQuadrants;
//////////////////////////////////////////////////////////////////////////
DEVICE_VARIABLE QuadrantEdges* g_dQuadrantsEdges;
//////////////////////////////////////////////////////////////////////////
DEVICE_VARIABLE unsigned char* g_dPopulationDensityMapData;
//////////////////////////////////////////////////////////////////////////
DEVICE_VARIABLE unsigned char* g_dWaterBodiesMapData;
//////////////////////////////////////////////////////////////////////////
DEVICE_VARIABLE unsigned char* g_dBlockadesMapData;
//////////////////////////////////////////////////////////////////////////
DEVICE_VARIABLE unsigned char* g_dNaturalPatternMapData;
//////////////////////////////////////////////////////////////////////////
DEVICE_VARIABLE unsigned char* g_dRadialPatternMapData;
//////////////////////////////////////////////////////////////////////////
DEVICE_VARIABLE unsigned char* g_dRasterPatternMapData;
//////////////////////////////////////////////////////////////////////////
DEVICE_VARIABLE Primitive* g_dPrimitives;

//////////////////////////////////////////////////////////////////////////
#define allocateAndInitializeImageMap(__name1, __name2) \
	if (__name1##Map.data != 0) \
	{ \
		unsigned int mapSize = __name1##Map.width * __name1##Map.height; \
		SAFE_MALLOC_ON_DEVICE(g_d##__name2##MapData, unsigned char, mapSize); \
		BIND_AS_TEXTURE2D(g_d##__name2##MapData, __name1##Texture, __name1##Map.width, __name1##Map.height); \
		MEMCPY_HOST_TO_DEVICE(g_d##__name2##MapData, __name1##Map.data, sizeof(unsigned char) * mapSize); \
		SAFE_MALLOC_ON_DEVICE(g_d##__name2##Map, ImageMap, 1); \
	}

#define deallocateImageMap(__name1, __name2) \
	if (__name1##Map.data != 0) \
	{ \
		UNBIND_TEXTURE2D(__name1##Texture); \
		SAFE_FREE_ON_DEVICE(g_d##__name2##MapData); \
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
	MEMCPY_HOST_TO_DEVICE(g_dQuadrants, graph->quadtree->quadrants, sizeof(Quadrant) * graph->quadtree->totalNumQuadrants);
	MEMCPY_HOST_TO_DEVICE(g_dQuadrantsEdges, graph->quadtree->quadrantsEdges, sizeof(QuadrantEdges) * graph->quadtree->numLeafQuadrants);
#ifdef COLLECT_STATISTICS
	INVOKE_GLOBAL_CODE10(updateNonPointerFields, 1, 1, g_dQuadtree, (int)graph->quadtree->numQuadrantEdges, graph->quadtree->worldBounds, graph->quadtree->maxDepth, graph->quadtree->maxQuadrants, graph->quadtree->totalNumQuadrants, graph->quadtree->numLeafQuadrants, (unsigned long)graph->quadtree->numCollisionChecks, (unsigned int)graph->quadtree->maxEdgesPerQuadrantInUse, (unsigned int)graph->quadtree->maxResultsPerQueryInUse);
#else
	INVOKE_GLOBAL_CODE7(updateNonPointerFields, 1, 1, g_dQuadtree, (int)graph->quadtree->numQuadrantEdges, graph->quadtree->worldBounds, graph->quadtree->maxDepth, graph->quadtree->maxQuadrants, graph->quadtree->totalNumQuadrants, graph->quadtree->numLeafQuadrants);
#endif
	MEMCPY_HOST_TO_DEVICE(g_dVertices, graph->vertices, sizeof(Vertex) * graph->numVertices);
	MEMCPY_HOST_TO_DEVICE(g_dEdges, graph->edges, sizeof(Edge) * graph->numEdges);
#ifdef COLLECT_STATISTICS
	INVOKE_GLOBAL_CODE4(updateNonPointerFields, 1, 1, g_dGraph, (int)graph->numVertices, (int)graph->numEdges, (unsigned long)graph->numCollisionChecks);
#else
	INVOKE_GLOBAL_CODE3(updateNonPointerFields, 1, 1, g_dGraph, (int)graph->numVertices, (int)graph->numEdges);
#endif
}

//////////////////////////////////////////////////////////////////////////
void RoadNetworkGraphGenerator::copyGraphToHost(Graph* graph)
{
	MEMCPY_DEVICE_TO_HOST(graph->quadtree->quadrants, g_dQuadrants, sizeof(Quadrant) * configuration.maxQuadrants);
	MEMCPY_DEVICE_TO_HOST(graph->quadtree->quadrantsEdges, g_dQuadrantsEdges, sizeof(QuadrantEdges) * configuration.maxQuadrants);

	Quadrant* quadrants = graph->quadtree->quadrants;
	QuadrantEdges* quadrantsEdges = graph->quadtree->quadrantsEdges;

	MEMCPY_DEVICE_TO_HOST(graph->quadtree, g_dQuadtree, sizeof(QuadTree));

	graph->quadtree->quadrants = quadrants;
	graph->quadtree->quadrantsEdges = quadrantsEdges;

	MEMCPY_DEVICE_TO_HOST(graph->vertices, g_dVertices, sizeof(Vertex) * configuration.maxVertices);
	MEMCPY_DEVICE_TO_HOST(graph->edges, g_dEdges, sizeof(Edge) * configuration.maxEdges);

	QuadTree* quadtree = graph->quadtree;
	Vertex* vertices = graph->vertices;
	Edge* edges = graph->edges;

	MEMCPY_DEVICE_TO_HOST(graph, g_dGraph, sizeof(Graph));

	graph->quadtree = quadtree;
	graph->vertices = vertices;
	graph->edges = edges;
}

//////////////////////////////////////////////////////////////////////////
void RoadNetworkGraphGenerator::execute()
{
	CREATE_GPU_TIMER(PrimaryRoadNetworkExpansion);
	CREATE_GPU_TIMER(SecondaryRoadNetworkExpansion);
	CREATE_GPU_TIMER(GraphMemoryCopy_GpuToCpu);
	CREATE_CPU_TIMER(GraphMemoryCopy_CpuToGpu);
	CREATE_GPU_TIMER(CollisionsComputation);
	CREATE_CPU_TIMER(PrimitivesExtraction);

	allocateAndInitializeImageMap(populationDensity, PopulationDensity);
	allocateAndInitializeImageMap(waterBodies, WaterBodies);
	allocateAndInitializeImageMap(blockades, Blockades);
	allocateAndInitializeImageMap(naturalPattern, NaturalPattern);
	allocateAndInitializeImageMap(radialPattern, RadialPattern);
	allocateAndInitializeImageMap(rasterPattern, RasterPattern);

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
	
	initializeGraphOnHost(graph, configuration.snapRadius, configuration.maxVertices, configuration.maxEdges, vertices, edges, quadtree);
	INVOKE_GLOBAL_CODE7(initializeGraphOnDevice, 1, 1, g_dGraph, configuration.snapRadius, configuration.maxVertices, configuration.maxEdges, g_dVertices, g_dEdges, g_dQuadtree);

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

	SAFE_MALLOC_ON_DEVICE(g_dPrimitives, Primitive, configuration.maxPrimitives);
	SAFE_MALLOC_ON_DEVICE(g_dConfiguration, Configuration, 1);

	START_CPU_TIMER(GraphMemoryCopy_CpuToGpu);

	copyGraphToDevice(graph);

	MEMCPY_HOST_TO_DEVICE(g_dWorkQueues1, workQueues1, sizeof(WorkQueue) * NUM_PROCEDURES);
	MEMCPY_HOST_TO_DEVICE(g_dWorkQueues2, workQueues2, sizeof(WorkQueue) * NUM_PROCEDURES);
	
	MEMCPY_HOST_TO_DEVICE(g_dConfiguration, const_cast<Configuration*>(&configuration), sizeof(Configuration));

	STOP_CPU_TIMER(GraphMemoryCopy_CpuToGpu);

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

	START_GPU_TIMER(PrimaryRoadNetworkExpansion);

	// expand primary road network
	expand(configuration.maxHighwayDerivation, 0, 3);

	STOP_GPU_TIMER(PrimaryRoadNetworkExpansion);

	float elapsedTime;
	GET_GPU_TIMER_ELAPSED_TIME(PrimaryRoadNetworkExpansion, elapsedTime);
	std::cout << "Primary Road Network Expansion: " << elapsedTime << " (ms)" << std::endl;

	START_GPU_TIMER(CollisionsComputation);

	coalesce();

	STOP_GPU_TIMER(CollisionsComputation);

	GET_GPU_TIMER_ELAPSED_TIME(CollisionsComputation, elapsedTime);
	std::cout << "Collisions Computation: " << elapsedTime << " (ms)" << std::endl;

	START_GPU_TIMER(GraphMemoryCopy_GpuToCpu);

	copyGraphToHost(graph);

	STOP_GPU_TIMER(GraphMemoryCopy_GpuToCpu);

	BaseGraph* graphCopy;
	Vertex* verticesCopy;
	Edge* edgesCopy;

	SAFE_MALLOC_ON_HOST(graphCopy, BaseGraph, 1);
	SAFE_MALLOC_ON_HOST(verticesCopy, Vertex, graph->numVertices);
	SAFE_MALLOC_ON_HOST(edgesCopy, Edge, graph->numEdges);

	graphCopy->vertices = verticesCopy;
	graphCopy->edges = edgesCopy;

	memcpy(graphCopy->vertices, graph->vertices, sizeof(Vertex) * graph->numVertices);
	memcpy(graphCopy->edges, graph->edges, sizeof(Edge) * graph->numEdges);
	
	graphCopy->numVertices = graph->numVertices;
	graphCopy->numEdges = graph->numEdges;

	Primitive* primitives;
	
	SAFE_MALLOC_ON_HOST(primitives, Primitive, configuration.maxPrimitives);

	memset(primitives, 0, sizeof(Primitive) * configuration.maxPrimitives);

	// extract the city cells

	START_CPU_TIMER(PrimitivesExtraction);

	unsigned int numPrimitives = extractPrimitives(graphCopy, primitives, configuration.maxPrimitives);

	STOP_CPU_TIMER(PrimitivesExtraction);

	GET_CPU_TIMER_ELAPSED_TIME(PrimitivesExtraction, elapsedTime);
	std::cout << "Primitives Extraction: " << elapsedTime << " (ms)" << std::endl;

	free(graphCopy);
	free(verticesCopy);
	free(edgesCopy);

	for (unsigned int i = 0; i < NUM_PROCEDURES; i++)
	{
		workQueues1[i].clear();
		workQueues2[i].clear();
	}

	// set street spawn points
	for (unsigned int i = 0; i < numPrimitives; i++)
	{
		Primitive& primitive = primitives[i];

		if (primitive.type != MINIMAL_CYCLE)
		{
			continue;
		}

		for (unsigned int j = 0; j < primitive.numEdges; j++)
		{
			Edge& edge = graph->edges[primitive.edges[j]];

			// FIXME: checking invariants
			if (edge.numPrimitives > 2)
			{
				THROW_EXCEPTION("edge.numPrimitives > 2");
			}

			edge.primitives[edge.numPrimitives++] = i;
		}

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

	START_CPU_TIMER(GraphMemoryCopy_CpuToGpu);
	
	MEMCPY_HOST_TO_DEVICE(g_dPrimitives, primitives, sizeof(Primitive) * numPrimitives);

	copyGraphToDevice(graph);

	MEMCPY_HOST_TO_DEVICE(g_dWorkQueues1, workQueues1, sizeof(WorkQueue) * NUM_PROCEDURES);
	MEMCPY_HOST_TO_DEVICE(g_dWorkQueues2, workQueues2, sizeof(WorkQueue) * NUM_PROCEDURES);

	STOP_CPU_TIMER(GraphMemoryCopy_CpuToGpu);

	START_GPU_TIMER(SecondaryRoadNetworkExpansion);

	// expand secondary road network
	expand(configuration.maxStreetDerivation, 3, 3);

	STOP_GPU_TIMER(SecondaryRoadNetworkExpansion);

	GET_GPU_TIMER_ELAPSED_TIME(SecondaryRoadNetworkExpansion, elapsedTime);
	std::cout << "Secondary Road Network Expansion: " << (elapsedTime + 500) << " (ms)" << std::endl;

	START_GPU_TIMER(GraphMemoryCopy_GpuToCpu);

	copyGraphToHost(graph);
	MEMCPY_DEVICE_TO_HOST(primitives, g_dPrimitives, sizeof(Primitive) * configuration.maxPrimitives);

	STOP_GPU_TIMER(GraphMemoryCopy_GpuToCpu);

	GET_GPU_TIMER_ELAPSED_TIME(GraphMemoryCopy_GpuToCpu, elapsedTime);
	std::cout << "Graph Memory Copy (Gpu -> Cpu): " << elapsedTime << " (ms)" << std::endl;

	GET_CPU_TIMER_ELAPSED_TIME(GraphMemoryCopy_CpuToGpu, elapsedTime);
	std::cout << "Graph Memory Copy (Cpu -> Gpu): " << elapsedTime << " (ms)" << std::endl;

#ifdef COLLECT_STATISTICS
	maxPrimitiveSize = 0;
	for (unsigned int i = 0; i < numPrimitives; i++)
	{
		maxPrimitiveSize = MathExtras::max(maxPrimitiveSize, primitives[i].numEdges);
	}

	std::cout << "vertices (allocated/in use): " << graph->maxVertices << " / " << graph->numVertices << std::endl;
	std::cout << "edges (allocated/in use): " << graph->maxEdges << " / " << graph->numEdges << std::endl;
	std::cout << "vertex in connections (max./max. in use): " << MAX_VERTEX_IN_CONNECTIONS << " / " << getMaxVertexInConnectionsInUse(graph) << std::endl;
	std::cout << "vertex out connections (max./max. in use): " << MAX_VERTEX_OUT_CONNECTIONS << " / " << getMaxVertexOutConnectionsInUse(graph) << std::endl;
	std::cout << "avg. vertex in connections in use: " << getAverageVertexInConnectionsInUse(graph) << std::endl;
	std::cout << "avg. vertex out connections in use: " << getAverageVertexOutConnectionsInUse(graph) << std::endl;
	std::cout << "num. primitives (max./in use): " << configuration.maxPrimitives << " / " << numPrimitives << std::endl;
	std::cout << "num. primitive edges (max./max. in use): " << MAX_EDGES_PER_PRIMITIVE << " / " << maxPrimitiveSize << std::endl;
	std::cout << "edges per quadrant (max./max. in use): " << MAX_EDGES_PER_QUADRANT << " / " << quadtree->maxEdgesPerQuadrantInUse << std::endl;
	std::cout << "memory (allocated/in use): " << toMegabytes(getAllocatedMemory(graph) + getAllocatedMemory(quadtree)) << " mb / " << toMegabytes(getMemoryInUse(graph) + getMemoryInUse(quadtree)) << " mb" << std::endl;
	std::cout << "num. collision checks: " << graph->numCollisionChecks + quadtree->numCollisionChecks << std::endl;
#endif

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
	deallocateImageMap(populationDensity, PopulationDensity);
	deallocateImageMap(waterBodies, WaterBodies);
	deallocateImageMap(blockades, Blockades);
	deallocateImageMap(naturalPattern, NaturalPattern);
	deallocateImageMap(radialPattern, RadialPattern);
	deallocateImageMap(rasterPattern, RasterPattern);
	SAFE_FREE_ON_DEVICE(g_dPrimitives);

	SAFE_FREE_ON_DEVICE(g_dQuadrants);
	SAFE_FREE_ON_DEVICE(g_dQuadrantsEdges);
	SAFE_FREE_ON_DEVICE(g_dQuadtree);
	SAFE_FREE_ON_DEVICE(g_dVertices);
	SAFE_FREE_ON_DEVICE(g_dEdges);
	SAFE_FREE_ON_DEVICE(g_dGraph);

	free(primitives);
	free(workQueues1);
	free(workQueues2);
	free(quadrants);
	free(quadrantsEdges);
	free(quadtree);
	free(graph);
	free(vertices);
	free(edges);

	DESTROY_GPU_TIMER(PrimaryRoadNetworkExpansion);
	DESTROY_GPU_TIMER(SecondaryRoadNetworkExpansion);
	DESTROY_GPU_TIMER(GraphMemoryCopy_GpuToCpu);
	DESTROY_CPU_TIMER(PrimitivesExtraction);
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void coalesceEdges(Graph* graph, QuadrantEdges* quadrantEdges)
{
	unsigned int i = THREAD_IDX_X;
	while (i < quadrantEdges->lastEdgeIndex)
	{
		Edge& thisEdge = graph->edges[quadrantEdges->edges[i]];

		for (unsigned int j = 0; j < quadrantEdges->lastEdgeIndex; j++)
		{
			Edge& otherEdge = graph->edges[quadrantEdges->edges[j]];

			if (thisEdge.index <= otherEdge.index)
			{
				continue;
			}

			bool tryAgain;
			do
			{
				tryAgain = false;
				vml_vec2 intersection;
				if (checkIntersection(graph, thisEdge, otherEdge, intersection))
				{
					tryAgain = true;
					if (ATOMIC_EXCH(thisEdge.owner, int, THREAD_IDX_X) == -1)
					{
						if (ATOMIC_EXCH(otherEdge.owner, int, THREAD_IDX_X) == -1)
						{
							int newVertexIndex = createVertex(graph, intersection);
							splitEdge(graph, otherEdge, newVertexIndex);
							splitEdge(graph, thisEdge, newVertexIndex);

							tryAgain = false;
							ATOMIC_EXCH(otherEdge.owner, int, -1);
						}

						ATOMIC_EXCH(thisEdge.owner, int, -1);	
					}
				}
#ifdef COLLECT_STATISTICS
				ATOMIC_ADD(graph->numCollisionChecks, unsigned int, 1);
#endif
			} while (tryAgain);
		}

		i += BLOCK_DIM_X;
	}
}

#ifdef USE_CUDA
//////////////////////////////////////////////////////////////////////////
__device__ volatile int g_dCounterLevel1;
//////////////////////////////////////////////////////////////////////////
__device__ volatile int g_dCounterLevel2;
//////////////////////////////////////////////////////////////////////////
__device__ volatile int g_dCounterLevel3;

//////////////////////////////////////////////////////////////////////////
__global__ void initializeCounters()
{
	g_dCounterLevel1 = g_dCounterLevel2 = g_dCounterLevel3 = 0;
}

//////////////////////////////////////////////////////////////////////////
__global__ void expansionKernel(unsigned int numDerivations, WorkQueue* workQueues1, WorkQueue* workQueues2, unsigned int startingQueue, unsigned int numQueues, Context* context)
{
	__shared__ WorkQueue* frontQueues;
	__shared__ WorkQueue* backQueues;
	__shared__ unsigned int reservedPops;
	__shared__ unsigned int head;
	__shared__ volatile unsigned int derivation;
	__shared__ volatile bool start;
	__shared__ volatile unsigned int run;
	__shared__ unsigned int currentQueue;

	if (threadIdx.x == 0)
	{
		frontQueues = workQueues1;
		backQueues = workQueues2;
		start = true;
		currentQueue = startingQueue + (blockIdx.x % numQueues);
	}

	__syncthreads();

	while (derivation < numDerivations)
	{
		if (start)
		{
			if (threadIdx.x == 0)
			{
				atomicAdd((int*)&g_dCounterLevel1, 1);
				atomicAdd((int*)&g_dCounterLevel2, 1);
				atomicAdd((int*)&g_dCounterLevel3, 1);
				run = 1;
			}

			__syncthreads();

			while (run > 0)
			{
				// block optimization
				if (threadIdx.x == 0)
				{
					frontQueues[currentQueue].reservePops(blockDim.x, &head, &reservedPops);

					/*unsigned int queueShifts = 0;
					// round-robin through all the queues until pops can be reserved
					while (reservedPops == 0 && ++queueShifts < numQueues)
					{
						currentQueue = startingQueue + ((currentQueue + 1) % numQueues);
						frontQueues[currentQueue].reservePops(blockDim.x, &head, &reservedPops);
					}*/
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
						atomicSub((int*)&g_dCounterLevel2, 1), run = 2;
					}
					else if (reservedPops == 0 && run == 2 && g_dCounterLevel2 == 0)
					{
						atomicSub((int*)&g_dCounterLevel3, 1), run = 3;
					}
					else if (reservedPops == 0 && run == 3 && g_dCounterLevel3 == 0)
					{
						run = 0;
					}
					else if (reservedPops != 0 && run != 1)
					{
						if (run == 2)
						{
							atomicAdd((int*)&g_dCounterLevel2, 1), run = 1;
						}

						if (run == 3)
						{
							atomicAdd((int*)&g_dCounterLevel2, 1), atomicAdd((int*)&g_dCounterLevel3, 1), run = 1;
						}
					}
				}

				__syncthreads();
			}
		}

		if (threadIdx.x == 0)
		{
			if (start)
			{
				atomicSub((int*)&g_dCounterLevel1, 1);
			}

			if (g_dCounterLevel1 == 0)
			{
				derivation++;
				WorkQueue* tmp = frontQueues;
				frontQueues = backQueues;
				backQueues = tmp;
				start = true;
			}
			else
			{
				start = false;
			}
		}

		__syncthreads();
	}
}

//////////////////////////////////////////////////////////////////////////
void RoadNetworkGraphGenerator::expand(unsigned int numDerivations, unsigned int startingQueue, unsigned int numQueues)
{
	initializeCounters<<<1, 1>>>();
	cudaCheckError();
	expansionKernel<<<NUM_EXPANSION_KERNEL_BLOCKS, NUM_EXPANSION_KERNEL_THREADS>>>(numDerivations, g_dWorkQueues1, g_dWorkQueues2, startingQueue, numQueues, g_dContext);
	cudaCheckError();
}

//////////////////////////////////////////////////////////////////////////
__global__ void coalescenceKernel(Graph* graph)
{
	__shared__ QuadrantEdges* quadrantEdges;

	if (threadIdx.x == 0)
	{
		quadrantEdges = &graph->quadtree->quadrantsEdges[blockIdx.x % graph->quadtree->numLeafQuadrants];
	}

	__syncthreads();

	coalesceEdges(graph, quadrantEdges);
}

//////////////////////////////////////////////////////////////////////////
void RoadNetworkGraphGenerator::coalesce()
{
	unsigned int numBlocks = MathExtras::pow(4u, configuration.quadtreeDepth) * NUM_COALESCE_KERNEL_BLOCKS_PER_QUADRANT;
	coalescenceKernel<<<numBlocks, NUM_COALESCE_KERNEL_THREADS>>>(g_dGraph);
	cudaCheckError();
}

#else
//////////////////////////////////////////////////////////////////////////
void expansionKernel(unsigned int numDerivations, WorkQueue* workQueues1, WorkQueue* workQueues2, unsigned int startingQueue, unsigned int numQueues, Context* context)
{
	unsigned int derivations = 0;
	WorkQueue* frontQueues = workQueues1;
	WorkQueue* backQueues = workQueues2;
	unsigned int currentQueue;

	while (derivations < numDerivations)
	{
		for (unsigned int i = 0, currentQueue = startingQueue; i < numQueues; i++, currentQueue = (startingQueue + (currentQueue + 1) % numQueues))
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

//////////////////////////////////////////////////////////////////////////
void RoadNetworkGraphGenerator::expand(unsigned int numDerivations, unsigned int startingQueue, unsigned int numQueues)
{
	expansionKernel(numDerivations, g_dWorkQueues1, g_dWorkQueues2, startingQueue, numQueues, g_dContext);
}

//////////////////////////////////////////////////////////////////////////
void coalescenceKernel(Graph* graph)
{
	QuadTree* quadtree = graph->quadtree;
	for (unsigned int i = 0; i < quadtree->numLeafQuadrants; i++)
	{
		coalesceEdges(graph, &quadtree->quadrantsEdges[i]);
	}
}

//////////////////////////////////////////////////////////////////////////
void RoadNetworkGraphGenerator::coalesce()
{
	coalescenceKernel(g_dGraph);
}

#endif