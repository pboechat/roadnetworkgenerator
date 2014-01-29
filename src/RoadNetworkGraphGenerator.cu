#include <RoadNetworkGraphGenerator.h>
#include <Procedures.h>
#include <Road.cuh>
#include <Branch.cuh>
#ifdef USE_QUADTREE
#include <Quadtree.cuh>
#endif
#include <WorkQueue.cuh>
#include <Primitive.h>
#include <BaseGraph.cuh>
#include <Graph.cuh>
#include <GlobalVariables.cuh>

#include <MathExtras.cuh>
#include <Box2D.cuh>
#include <MinimalCycleBasis.h>
#include <ConvexHull.h>
#include <OBB2D.h>

#include <vector_math.h>

//////////////////////////////////////////////////////////////////////////
//	DEVICE VARIABLES
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE WorkQueue g_dWorkQueues1[NUM_PROCEDURES];
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE WorkQueue g_dWorkQueues2[NUM_PROCEDURES];
//////////////////////////////////////////////////////////////////////////
WorkQueue g_hWorkQueues1[NUM_PROCEDURES];
//////////////////////////////////////////////////////////////////////////
WorkQueue g_hWorkQueues2[NUM_PROCEDURES];
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE RoadNetworkGraph::Vertex* g_dVertices;
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE RoadNetworkGraph::Edge* g_dEdges;
#ifdef USE_QUADTREE
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE RoadNetworkGraph::QuadTree* g_dQuadtree;
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE RoadNetworkGraph::Quadrant* g_dQuadrants;
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE RoadNetworkGraph::QuadrantEdges* g_dQuadrantsEdges;
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE RoadNetworkGraph::EdgeIndex* g_dQueryResults;
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
#define allocateAndInitializeImageMap(__name1, __name2) \
	if (__name1##Map.hasData()) \
	{ \
		unsigned int mapSize = __name1##Map.getWidth() * __name1##Map.getHeight(); \
		MALLOC_ON_DEVICE(g_d##__name2##MapData, unsigned char, mapSize); \
		MEMCPY_HOST_TO_DEVICE(g_d##__name2##MapData, __name1##Map.getData(), sizeof(unsigned char) * mapSize); \
		MALLOC_ON_DEVICE(g_d##__name2##Map, ImageMap, 1); \
	}

//////////////////////////////////////////////////////////////////////////
void RoadNetworkGraphGenerator::notifyObservers(RoadNetworkGraph::Graph* graph, unsigned int numPrimitives, RoadNetworkGraph::Primitive* primitives)
{
	for (unsigned int i = 0; i < observers.size(); i++)
	{
		observers[i]->update(graph, numPrimitives, primitives);
	}
}

//////////////////////////////////////////////////////////////////////////
void RoadNetworkGraphGenerator::copyGraphToDevice(RoadNetworkGraph::Graph* graph)
{
#ifdef USE_QUADTREE
	MEMCPY_HOST_TO_DEVICE(g_dQuadrants, graph->quadtree->quadrants, sizeof(RoadNetworkGraph::Quadrant) * configuration.maxQuadrants);
	MEMCPY_HOST_TO_DEVICE(g_dQuadrantsEdges, graph->quadtree->quadrantsEdges, sizeof(RoadNetworkGraph::QuadrantEdges) * configuration.maxQuadrants);
#ifdef _DEBUG
	//INVOKE_GLOBAL_CODE11(RoadNetworkGraph::updateNonPointerFields, 1, 1, g_dQuadtree, g_hQuadtree->numQuadrantEdges, g_hQuadtree->maxResultsPerQuery, g_hQuadtree->worldBounds, g_hQuadtree->maxDepth, g_hQuadtree->maxQuadrants, g_hQuadtree->totalNumQuadrants, g_hQuadtree->numLeafQuadrants, g_hQuadtree->numCollisionChecks, g_hQuadtree->maxEdgesPerQuadrantInUse, g_hQuadtree->maxResultsPerQueryInUse);
#else
	//INVOKE_GLOBAL_CODE8(RoadNetworkGraph::updateNonPointerFields, 1, 1, g_dQuadtree, g_hQuadtree->numQuadrantEdges, g_hQuadtree->maxResultsPerQuery, g_hQuadtree->worldBounds, g_hQuadtree->maxDepth, g_hQuadtree->maxQuadrants, g_hQuadtree->totalNumQuadrants, g_hQuadtree->numLeafQuadrants);
#endif
#endif

	MEMCPY_HOST_TO_DEVICE(g_dVertices, graph->vertices, sizeof(RoadNetworkGraph::Vertex) * configuration.maxVertices);
	MEMCPY_HOST_TO_DEVICE(g_dEdges, graph->edges, sizeof(RoadNetworkGraph::Edge) * configuration.maxEdges);
#ifdef _DEBUG
	//INVOKE_GLOBAL_CODE7(RoadNetworkGraph::updateNonPointerFields, 1, 1, g_dGraph, g_hGraph->numVertices, g_hGraph->numEdges, g_hGraph->maxVertices, g_hGraph->maxEdges, g_hGraph->maxResultsPerQuery, g_hGraph->numCollisionChecks);
#else
	//INVOKE_GLOBAL_CODE6(RoadNetworkGraph::updateNonPointerFields, 1, 1, g_dGraph, g_hGraph->numVertices, g_hGraph->numEdges, g_hGraph->maxVertices, g_hGraph->maxEdges, g_hGraph->maxResultsPerQuery);
#endif
}

//////////////////////////////////////////////////////////////////////////
void RoadNetworkGraphGenerator::copyGraphToHost(RoadNetworkGraph::Graph* graph)
{
#ifdef USE_QUADTREE
	MEMCPY_HOST_TO_DEVICE(graph->quadtree->quadrants, g_dQuadrants, sizeof(RoadNetworkGraph::Quadrant) * configuration.maxQuadrants);
	MEMCPY_HOST_TO_DEVICE(graph->quadtree->quadrantsEdges, g_dQuadrantsEdges, sizeof(RoadNetworkGraph::QuadrantEdges) * configuration.maxQuadrants);
	MEMCPY_DEVICE_TO_HOST(graph->quadtree, g_dQuadtree, sizeof(RoadNetworkGraph::QuadTree));
#endif

	MEMCPY_DEVICE_TO_HOST(graph->vertices, g_dVertices, sizeof(RoadNetworkGraph::Vertex) * configuration.maxVertices);
	MEMCPY_DEVICE_TO_HOST(graph->edges, g_dEdges, sizeof(RoadNetworkGraph::Edge) * configuration.maxEdges);
	MEMCPY_DEVICE_TO_HOST(graph, g_dGraph, sizeof(RoadNetworkGraph::Graph));
}

//////////////////////////////////////////////////////////////////////////
void RoadNetworkGraphGenerator::execute()
{
	allocateAndInitializeImageMap(populationDensity, PopulationDensity);
	allocateAndInitializeImageMap(waterBodies, WaterBodies);
	allocateAndInitializeImageMap(blockades, Blockades);
	allocateAndInitializeImageMap(naturalPattern, NaturalPattern);
	allocateAndInitializeImageMap(radialPattern, RadialPattern);
	allocateAndInitializeImageMap(rasterPattern, RasterPattern);

	MALLOC_ON_DEVICE(g_dPopulationDensitiesSamplingBuffer, unsigned char, configuration.samplingArc);
	MALLOC_ON_DEVICE(g_dDistancesSamplingBuffer, unsigned int, configuration.samplingArc);

#ifdef USE_QUADTREE
	RoadNetworkGraph::QuadTree* quadtree = (RoadNetworkGraph::QuadTree*)malloc(sizeof(RoadNetworkGraph::QuadTree));
	RoadNetworkGraph::EdgeIndex* queryResults = (RoadNetworkGraph::EdgeIndex*)malloc(sizeof(RoadNetworkGraph::EdgeIndex) * configuration.maxResultsPerQuery);
	RoadNetworkGraph::Quadrant* quadrants = (RoadNetworkGraph::Quadrant*)malloc(sizeof(RoadNetworkGraph::Quadrant) * configuration.maxQuadrants);
	RoadNetworkGraph::QuadrantEdges* quadrantsEdges = (RoadNetworkGraph::QuadrantEdges*)malloc(sizeof(RoadNetworkGraph::QuadrantEdges) * configuration.maxQuadrants);

	memset(quadrants, 0, sizeof(RoadNetworkGraph::Quadrant) * configuration.maxQuadrants);
	memset(quadrantsEdges, 0, sizeof(RoadNetworkGraph::QuadrantEdges) * configuration.maxQuadrants);

	MALLOC_ON_DEVICE(g_dQuadtree, RoadNetworkGraph::QuadTree, 1);
	MALLOC_ON_DEVICE(g_dQueryResults, RoadNetworkGraph::EdgeIndex, configuration.maxResultsPerQuery);
	MALLOC_ON_DEVICE(g_dQuadrants, RoadNetworkGraph::Quadrant, configuration.maxQuadrants);
	MALLOC_ON_DEVICE(g_dQuadrantsEdges, RoadNetworkGraph::QuadrantEdges, configuration.maxQuadrants);

	MEMSET_ON_DEVICE(g_dQuadrants, 0, sizeof(RoadNetworkGraph::Quadrant) * configuration.maxQuadrants);
	MEMSET_ON_DEVICE(g_dQuadrantsEdges, 0, sizeof(RoadNetworkGraph::QuadrantEdges) * configuration.maxQuadrants);

	Box2D worldBounds(0.0f, 0.0f, (float)configuration.worldWidth, (float)configuration.worldHeight);

	//INVOKE_GLOBAL_CODE7(RoadNetworkGraph::initializeQuadtreeOnDevice, 1, 1, g_dQuadtree, worldBounds, configuration.quadtreeDepth, configuration.maxResultsPerQuery, configuration.maxQuadrants, g_dQuadrants, g_dQuadrantsEdges);
	//RoadNetworkGraph::initializeQuadtreeOnHost(quadtree, worldBounds, configuration.quadtreeDepth, configuration.maxResultsPerQuery, configuration.maxQuadrants, quadrants, quadrantsEdges);
#endif

	RoadNetworkGraph::Graph* graph = (RoadNetworkGraph::Graph*)malloc(sizeof(RoadNetworkGraph::Graph));
	RoadNetworkGraph::Vertex* vertices = (RoadNetworkGraph::Vertex*)malloc(sizeof(RoadNetworkGraph::Vertex) * configuration.maxVertices);
	RoadNetworkGraph::Edge* edges = (RoadNetworkGraph::Edge*)malloc(sizeof(RoadNetworkGraph::Edge) * configuration.maxEdges);

	memset(vertices, 0, sizeof(RoadNetworkGraph::Vertex) * configuration.maxVertices);
	memset(edges, 0, sizeof(RoadNetworkGraph::Edge) * configuration.maxEdges);

	MALLOC_ON_DEVICE(g_dGraph, RoadNetworkGraph::Graph, 1);
	MALLOC_ON_DEVICE(g_dVertices, RoadNetworkGraph::Vertex, configuration.maxVertices);
	MALLOC_ON_DEVICE(g_dEdges, RoadNetworkGraph::Edge, configuration.maxEdges);

	MEMSET_ON_DEVICE(g_dVertices, 0, sizeof(RoadNetworkGraph::Vertex) * configuration.maxVertices);
	MEMSET_ON_DEVICE(g_dEdges, 0, sizeof(RoadNetworkGraph::Edge) * configuration.maxEdges);
	
#ifdef USE_QUADTREE
	RoadNetworkGraph::initializeGraphOnHost(graph, configuration.snapRadius, configuration.maxVertices, configuration.maxEdges, vertices, edges, quadtree, configuration.maxResultsPerQuery, queryResults);
	//INVOKE_GLOBAL_CODE9(RoadNetworkGraph::initializeGraphOnDevice, 1, 1, g_dGraph, configuration.snapRadius, configuration.maxVertices, configuration.maxEdges, g_dVertices, g_dEdges, g_dQuadtree, configuration.maxResultsPerQuery, g_dQueryResults);
#else
	RoadNetworkGraph::initializeGraphOnHost(g_hGraph, configuration.snapRadius, configuration.maxVertices, configuration.maxEdges, vertices, eEdges);
	//INVOKE_GLOBAL_CODE6(RoadNetworkGraph::initializeGraphOnDevice, 1, 1, g_dGraph, configuration.snapRadius, configuration.maxVertices, configuration.maxEdges, g_dVertices, g_dEdges);
#endif

	memset(g_hWorkQueues1, 0, sizeof(WorkQueue) * NUM_PROCEDURES);
	memset(g_hWorkQueues2, 0, sizeof(WorkQueue) * NUM_PROCEDURES);

	MEMSET_ON_DEVICE(g_dWorkQueues1, 0, sizeof(WorkQueue) * NUM_PROCEDURES);
	MEMSET_ON_DEVICE(g_dWorkQueues2, 0, sizeof(WorkQueue) * NUM_PROCEDURES);

	WorkQueue workQueues1[NUM_PROCEDURES];
	WorkQueue workQueues2[NUM_PROCEDURES];

	// set highway spawn points
	for (unsigned int i = 0; i < configuration.numSpawnPoints; i++)
	{
		vml_vec2 spawnPoint = configuration.spawnPoints[i];
		RoadNetworkGraph::VertexIndex source = RoadNetworkGraph::createVertex(graph, spawnPoint);
		workQueues1[EVALUATE_HIGHWAY].pushOnHost(Highway(0, RoadAttributes(source, configuration.highwayLength, 0), UNASSIGNED));
		workQueues1[EVALUATE_HIGHWAY].pushOnHost(Highway(0, RoadAttributes(source, configuration.highwayLength, -HALF_PI), UNASSIGNED));
		workQueues1[EVALUATE_HIGHWAY].pushOnHost(Highway(0, RoadAttributes(source, configuration.highwayLength, HALF_PI), UNASSIGNED));
		workQueues1[EVALUATE_HIGHWAY].pushOnHost(Highway(0, RoadAttributes(source, configuration.highwayLength, PI), UNASSIGNED));
	}
	copyGraphToDevice(graph);

	MEMCPY_HOST_TO_DEVICE(g_dWorkQueues1, workQueues1, sizeof(WorkQueue) * NUM_PROCEDURES);
	MEMCPY_HOST_TO_DEVICE(g_dWorkQueues2, workQueues2, sizeof(WorkQueue) * NUM_PROCEDURES);

	expand(configuration.maxHighwayDerivation);

	copyGraphToHost(graph);

	RoadNetworkGraph::BaseGraph* graphCopy = (RoadNetworkGraph::BaseGraph*)malloc(sizeof(RoadNetworkGraph::BaseGraph));
	RoadNetworkGraph::Vertex* verticesCopy = (RoadNetworkGraph::Vertex*)malloc(sizeof(RoadNetworkGraph::Vertex) * configuration.maxVertices);
	RoadNetworkGraph::Edge* edgesCopy = (RoadNetworkGraph::Edge*)malloc(sizeof(RoadNetworkGraph::Edge) * configuration.maxEdges);

	graphCopy->vertices = verticesCopy;
	graphCopy->edges = edgesCopy;

	memcpy(graphCopy->vertices, graph->vertices, sizeof(RoadNetworkGraph::Vertex) * configuration.maxVertices);
	memcpy(graphCopy->edges, graph->edges, sizeof(RoadNetworkGraph::Edge) * configuration.maxEdges);
	
	graphCopy->numVertices = graph->numVertices;
	graphCopy->numEdges = graph->numEdges;

	RoadNetworkGraph::Primitive* primitives = (RoadNetworkGraph::Primitive*)malloc(sizeof(RoadNetworkGraph::Primitive) * configuration.maxPrimitives);
	memset(primitives, 0, sizeof(RoadNetworkGraph::Primitive) * configuration.maxPrimitives);

	// extract the allotments from graph copy
	RoadNetworkGraph::allocateExtractionBuffers(configuration.maxVertices, configuration.maxEdgeSequences, configuration.maxVisitedVertices);
	unsigned int numPrimitives = RoadNetworkGraph::extractPrimitives(graphCopy, primitives, configuration.maxPrimitives);
	RoadNetworkGraph::freeExtractionBuffers();

	free(graphCopy);
	free(verticesCopy);
	free(edgesCopy);

	//clearWorkQueuesOnHost();

	maxPrimitiveSize = 0;
	// set street spawn points
	for (unsigned int i = 0; i < numPrimitives; i++)
	{
		const RoadNetworkGraph::Primitive& primitive = primitives[i];

		if (primitive.type != RoadNetworkGraph::MINIMAL_CYCLE)
		{
			continue;
		}

		maxPrimitiveSize = MathExtras::max<unsigned int>(maxPrimitiveSize, primitive.numVertices);

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

		RoadNetworkGraph::VertexIndex source = RoadNetworkGraph::createVertex(graph, centroid);
		workQueues1[EVALUATE_STREET].pushOnHost(Street(0, RoadAttributes(source, configuration.streetLength, angle), UNASSIGNED));
		workQueues1[EVALUATE_STREET].pushOnHost(Street(0, RoadAttributes(source, configuration.streetLength, -HALF_PI + angle), UNASSIGNED));
		workQueues1[EVALUATE_STREET].pushOnHost(Street(0, RoadAttributes(source, configuration.streetLength, HALF_PI + angle), UNASSIGNED));
		workQueues1[EVALUATE_STREET].pushOnHost(Street(0, RoadAttributes(source, configuration.streetLength, PI + angle), UNASSIGNED));
	}
	copyGraphToDevice(graph);

	MEMCPY_HOST_TO_DEVICE(g_dWorkQueues1, workQueues1, sizeof(WorkQueue) * NUM_PROCEDURES);
	MEMCPY_HOST_TO_DEVICE(g_dWorkQueues2, workQueues2, sizeof(WorkQueue) * NUM_PROCEDURES);

	expand(configuration.maxHighwayDerivation);

	copyGraphToHost(graph);

	notifyObservers(graph, numPrimitives, primitives);

	FREE_ON_DEVICE(g_dPopulationDensityMap);
	FREE_ON_DEVICE(g_dWaterBodiesMap);
	FREE_ON_DEVICE(g_dBlockadesMap);
	FREE_ON_DEVICE(g_dNaturalPatternMap);
	FREE_ON_DEVICE(g_dRadialPatternMap);
	FREE_ON_DEVICE(g_dRasterPatternMap);
	FREE_ON_DEVICE(g_dPopulationDensityMapData);
	FREE_ON_DEVICE(g_dWaterBodiesMapData);
	FREE_ON_DEVICE(g_dBlockadesMapData);
	FREE_ON_DEVICE(g_dNaturalPatternMapData);
	FREE_ON_DEVICE(g_dRadialPatternMapData);
	FREE_ON_DEVICE(g_dRasterPatternMapData);

	FREE_ON_DEVICE(g_dPopulationDensitiesSamplingBuffer);
	FREE_ON_DEVICE(g_dDistancesSamplingBuffer);

#ifdef USE_QUADTREE
	FREE_ON_DEVICE(g_dQuadrants);
	FREE_ON_DEVICE(g_dQuadrantsEdges);
	FREE_ON_DEVICE(g_dQueryResults);
	FREE_ON_DEVICE(g_dQuadtree);
#endif
	FREE_ON_DEVICE(g_dVertices);
	FREE_ON_DEVICE(g_dEdges);
	FREE_ON_DEVICE(g_dGraph);

#ifdef USE_QUADTREE
	free(quadrants);
	free(quadrantsEdges);
	free(queryResults);
	free(quadtree);
#endif
	free(graph);
	free(vertices);
	free(edges);
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
__global__ void kernel(unsigned int numDerivations);

void RoadNetworkGraphGenerator::expand(unsigned int numDerivations)
{
	initializeKernel<<<1, 1>>>();
	cudaCheckError();
	kernel<<<NUM_PROCEDURES, NUM_THREADS>>>(g_dConfiguration->maxStreetDerivation);
	cudaCheckError();
}

//////////////////////////////////////////////////////////////////////////
__global__ void kernel(unsigned int numDerivations)
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
		frontQueues = g_dWorkQueues1;
		backQueues = g_dWorkQueues2;
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
						EvaluateHighwayBranch::execute(highwayBranch, backQueues);
					}
					break;
				case EVALUATE_HIGHWAY:
					{
						Highway highway;
						frontQueues[EVALUATE_HIGHWAY].popReserved(head + threadIdx.x, highway);
						EvaluateHighway::execute(highway, backQueues);
					}
					break;
				case INSTANTIATE_HIGHWAY:
					{
						Highway highway;
						frontQueues[INSTANTIATE_HIGHWAY].popReserved(head + threadIdx.x, highway);
						InstantiateHighway::execute(highway, backQueues);
					}
					break;
				case EVALUATE_STREET_BRANCH:
					{
						StreetBranch streetBranch;
						frontQueues[EVALUATE_STREET_BRANCH].popReserved(head + threadIdx.x, streetBranch);
						EvaluateStreetBranch::execute(streetBranch, backQueues);
					}
					break;
				case EVALUATE_STREET:
					{
						Street street;
						frontQueues[EVALUATE_STREET].popReserved(head + threadIdx.x, street);
						EvaluateStreet::execute(street, backQueues);
					}
					break;
				case INSTANTIATE_STREET:
					{
						Street street;
						frontQueues[INSTANTIATE_STREET].popReserved(head + threadIdx.x, street);
						InstantiateStreet::execute(street, backQueues);
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
void RoadNetworkGraphGenerator::expand(unsigned int numDerivations)
{
	unsigned int derivations = 0;
	WorkQueue* frontQueues = g_dWorkQueues1;
	WorkQueue* backQueues = g_dWorkQueues2;

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
						frontQueues[EVALUATE_HIGHWAY_BRANCH].pop(highwayBranch);
						EvaluateHighwayBranch::execute(highwayBranch, backQueues);
					}
				}
				break;
			case EVALUATE_HIGHWAY:
				{
					Highway highway;
					while (frontQueues[EVALUATE_HIGHWAY].count > 0)
					{
						frontQueues[EVALUATE_HIGHWAY].pop(highway);
						EvaluateHighway::execute(highway, backQueues);
					}
				}
				break;
			case INSTANTIATE_HIGHWAY:
				{
					Highway highway;
					while (frontQueues[INSTANTIATE_HIGHWAY].count > 0)
					{
						frontQueues[INSTANTIATE_HIGHWAY].pop(highway);
						InstantiateHighway::execute(highway, backQueues);
					}
				}
				break;
			case EVALUATE_STREET_BRANCH:
				{
					StreetBranch streetBranch;
					while (frontQueues[EVALUATE_STREET_BRANCH].count > 0)
					{
						frontQueues[EVALUATE_STREET_BRANCH].pop(streetBranch);
						EvaluateStreetBranch::execute(streetBranch, backQueues);
					}
				}
				break;
			case EVALUATE_STREET:
				{
					Street street;
					while (frontQueues[EVALUATE_STREET].count > 0)
					{
						frontQueues[EVALUATE_STREET].pop(street);
						EvaluateStreet::execute(street, backQueues);
					}
				}
				break;
			case INSTANTIATE_STREET:
				{
					Street street;
					while (frontQueues[INSTANTIATE_STREET].count > 0)
					{
						frontQueues[INSTANTIATE_STREET].pop(street);
						InstantiateStreet::execute(street, backQueues);
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