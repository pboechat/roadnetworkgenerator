// memory leak detection
#include <vld.h>

#include "Defines.h"
#include <RoadNetworkInputController.h>
#include <SceneRenderer.h>
#include <RoadNetworkGeometry.h>
#include <RoadNetworkLabels.h>
#include <RoadNetworkGenerator.h>
#include <Configuration.h>
#include <Graph.h>
#include <QuadTree.h>
#include <Globals.h>

#include <Application.h>
#include <Camera.h>
#include <Box2D.h>
#include <Timer.h>
#include <MathExtras.h>

#include <vector_math.h>

#include <string>
#include <iostream>
#include <io.h>
#include <iomanip>

void printUsage()
{
	std::cerr << "Command line options: <width> <height> <configuration file>";
}

void generateAndDisplay(const std::string& configurationFile, SceneRenderer& renderer, RoadNetworkGeometry& geometry, RoadNetworkLabels& labels, Camera& camera)
{
	//////////////////////////////////////////////////////////////////////////
	//	ALLOCATE CONFIGURATION
	//////////////////////////////////////////////////////////////////////////
	if (g_dConfiguration != 0)
	{
		FREE_ON_DEVICE(g_dConfiguration);
		g_dConfiguration = 0;
	}

	g_hConfiguration = (Configuration*)malloc(sizeof(Configuration));
	g_hConfiguration->loadFromFile(configurationFile);

	MALLOC_ON_DEVICE(g_dConfiguration, Configuration, 1);
	MEMCPY_HOST_TO_DEVICE(g_dConfiguration, g_hConfiguration, sizeof(Configuration));

	allocateAndInitializeImageMaps(g_hConfiguration->populationDensityMapFilePath,
								   g_hConfiguration->waterBodiesMapFilePath,
								   g_hConfiguration->blockadesMapFilePath,
								   g_hConfiguration->naturalPatternMapFilePath,
								   g_hConfiguration->radialPatternMapFilePath,
								   g_hConfiguration->rasterPatternMapFilePath,
								   g_hConfiguration->worldWidth,
								   g_hConfiguration->worldHeight);

	allocateSamplingBuffers(g_hConfiguration->samplingArc);
	allocateWorkQueues(g_hConfiguration->maxWorkQueueCapacity);

	Box2D worldBounds(0.0f, 0.0f, (float)g_hConfiguration->worldWidth, (float)g_hConfiguration->worldHeight);

	// TODO:
	renderer.setUpImageMaps(worldBounds, g_dPopulationDensityMap, g_dWaterBodiesMap, g_dBlockadesMap);

	//////////////////////////////////////////////////////////////////////////
	//	ALLOCATE GRAPH
	//////////////////////////////////////////////////////////////////////////
	if (g_dGraph != 0)
	{
		FREE_ON_DEVICE(g_dGraph);
		g_dGraph = 0;
	}

	MALLOC_ON_DEVICE(g_dGraph, RoadNetworkGraph::Graph, 1);

	if (g_hGraphCopy != 0)
	{
		free(g_hGraphCopy);
		g_hGraphCopy = 0;
	}

	if (g_hVerticesCopy != 0)
	{
		free(g_hVerticesCopy);
		g_hVerticesCopy = 0;
	}

	if (g_hEdgesCopy != 0)
	{
		free(g_hEdgesCopy);
		g_hEdgesCopy = 0;
	}

	g_hGraphCopy = (RoadNetworkGraph::BaseGraph*)malloc(sizeof(RoadNetworkGraph::BaseGraph));
	g_hVerticesCopy = (RoadNetworkGraph::Vertex*)malloc(sizeof(RoadNetworkGraph::Vertex) * g_hConfiguration->maxVertices);
	g_hEdgesCopy = (RoadNetworkGraph::Edge*)malloc(sizeof(RoadNetworkGraph::Edge) * g_hConfiguration->maxEdges);

	allocateGraphBuffers(g_hConfiguration->maxVertices, g_hConfiguration->maxEdges);
#ifdef USE_QUADTREE
	//////////////////////////////////////////////////////////////////////////
	//	ALLOCATE QUADTREE
	//////////////////////////////////////////////////////////////////////////

	if (g_dQuadtree != 0)
	{
		FREE_ON_DEVICE(g_dQuadtree);
		g_dQuadtree = 0;
	}

	MALLOC_ON_DEVICE(g_dQuadtree, RoadNetworkGraph::QuadTree, 1);

	allocateQuadtreeBuffers(g_hConfiguration->maxResultsPerQuery, g_hConfiguration->maxQuadrants);

	INVOKE_GLOBAL_CODE7(RoadNetworkGraph::initializeQuadtree, 1, 1, g_dQuadtree, worldBounds, g_hConfiguration->quadtreeDepth, g_hConfiguration->maxResultsPerQuery, g_hConfiguration->maxQuadrants, g_dQuadrants, g_dQuadrantsEdges);
	INVOKE_GLOBAL_CODE9(RoadNetworkGraph::initializeGraph, 1, 1, g_dGraph, g_hConfiguration->snapRadius, g_hConfiguration->maxVertices, g_hConfiguration->maxEdges, g_dVertices, g_dEdges, g_dQuadtree, g_hConfiguration->maxResultsPerQuery, g_dQueryResults);
#else
	INVOKE_GLOBAL_CODE6(RoadNetworkGraph::initializeGraph, 1, 1, g_dGraph, g_hConfiguration->snapRadius, g_hConfiguration->maxVertices, g_hConfiguration->maxEdges, g_dVertices, g_dEdges);
#endif

	//////////////////////////////////////////////////////////////////////////
	//	ALLOCATE PRIMITIVES
	//////////////////////////////////////////////////////////////////////////

	if (g_hPrimitives != 0)
	{
		free(g_hPrimitives);
		g_hPrimitives = 0;
	}

	g_hPrimitives = (RoadNetworkGraph::Primitive*)malloc(sizeof(RoadNetworkGraph::Primitive) * g_hConfiguration->maxPrimitives);
	memset(g_hPrimitives, 0, sizeof(RoadNetworkGraph::Primitive) * g_hConfiguration->maxPrimitives);

	RoadNetworkGenerator generator;

#ifdef _DEBUG
	Timer timer;
	timer.start();
#endif
	generator.execute();

	// TODO:
	RoadNetworkGraph::Graph* graph = (RoadNetworkGraph::Graph*)malloc(sizeof(RoadNetworkGraph::Graph));
	MEMCPY_DEVICE_TO_HOST(graph, g_dGraph, sizeof(RoadNetworkGraph::Graph));

	RoadNetworkGraph::QuadTree* quadtree = (RoadNetworkGraph::QuadTree*)malloc(sizeof(RoadNetworkGraph::QuadTree));
	MEMCPY_DEVICE_TO_HOST(quadtree, g_dQuadtree, sizeof(RoadNetworkGraph::QuadTree));

	// TODO:
	graph->quadtree = quadtree;

#ifdef _DEBUG
	timer.end();
	std::cout << "*****************************" << std::endl;
	std::cout << "	DETAILS:				   " << std::endl;
	std::cout << "*****************************" << std::endl;
#ifdef _DEBUG
	std::cout << "seed: " << g_dConfiguration->seed << std::endl;
#endif
	std::cout << "generation time: " << timer.elapsedTime() << " seconds" << std::endl;
	std::cout << "last highway derivation (max./real): " << g_hConfiguration->maxHighwayDerivation << " / " << generator.getLastHighwayDerivation() << std::endl;
	std::cout << "last street derivation (max./real): " << g_hConfiguration->maxStreetDerivation << " / " << generator.getLastStreetDerivation() << std::endl;
	std::cout << "work queue capacity (max/max. in use): " << (g_hConfiguration->maxWorkQueueCapacity * NUM_PROCEDURES) << " / " << generator.getMaxWorkQueueCapacityUsed() << std::endl;
	std::cout << "memory (allocated/in use): " << toMegabytes(getAllocatedMemory(graph)) << " MB / " << toMegabytes(getMemoryInUse(graph)) << " MB" << std::endl;
	std::cout << "vertices (allocated/in use): " << getAllocatedVertices(graph) << " / " << getVerticesInUse(graph) << std::endl;
	std::cout << "edges (allocated/in use): " << getAllocatedEdges(graph) << " / " << getEdgesInUse(graph) << std::endl;
	std::cout << "vertex in connections (max./max. in use): " << getMaxVertexInConnections(graph) << " / " << getMaxVertexInConnectionsInUse(graph) << std::endl;
	std::cout << "vertex out connections (max./max. in use): " << getMaxVertexOutConnections(graph) << " / " << getMaxVertexOutConnectionsInUse(graph) << std::endl;
	std::cout << "avg. vertex in connections in use: " << getAverageVertexInConnectionsInUse(graph) << std::endl;
	std::cout << "avg. vertex out connections in use: " << getAverageVertexOutConnectionsInUse(graph) << std::endl;
#ifdef USE_QUADTREE
	std::cout << "edges per quadrant (max./max. in use): " << MAX_EDGES_PER_QUADRANT << " / " << getMaxEdgesPerQuadrantInUse(quadtree) << std::endl;
	std::cout << "results per query (max./max. in use): " << g_hConfiguration->maxResultsPerQuery << " / " << getMaxResultsPerQueryInUse(quadtree) << std::endl;
#endif
	std::cout << "primitive size (max./max. in use): " << MAX_VERTICES_PER_PRIMITIVE << "/" << generator.getMaxPrimitiveSize() << std::endl;
	std::cout << "num. collision checks: " << getNumCollisionChecks(graph) << std::endl;
	std::cout  << std::endl << std::endl;
#endif

	//////////////////////////////////////////////////////////////////////////
	//	ALLOCATE VBOs
	//////////////////////////////////////////////////////////////////////////

	allocateGraphicsBuffers(g_hConfiguration->vertexBufferSize, g_hConfiguration->indexBufferSize);

	// TODO:
	geometry.build();
	labels.build();

	camera.centerOnTarget(worldBounds);

	free(quadtree);
	free(graph);
}

int main(int argc, char** argv)
{
	int returnValue = -1;

	try
	{
		if (argc < 4)
		{
			printUsage();
			exit(EXIT_FAILURE);
		}

		unsigned int screenWidth = (unsigned int)atoi(argv[1]);
		unsigned int screenHeight = (unsigned int)atoi(argv[2]);
		std::string configurationFile = argv[3];

		if (configurationFile.empty())
		{
			printUsage();
			exit(EXIT_FAILURE);
		}

		Application application("Road Network Generator (CPU)", screenWidth, screenHeight);

		if (gl3wInit())
		{
			throw std::runtime_error("gl3wInit() failed");
		}

		INITIALIZE_DEVICE();

		Camera camera(screenWidth, screenHeight, FOVY_DEG, ZNEAR, ZFAR);
		RoadNetworkGeometry geometry;
		RoadNetworkLabels labels;
		SceneRenderer renderer(camera, geometry, labels);
		RoadNetworkInputController inputController(camera, configurationFile, renderer, geometry, labels, generateAndDisplay);
		application.setCamera(camera);
		application.setRenderer(renderer);
		application.setInputController(inputController);
		generateAndDisplay(configurationFile, renderer, geometry, labels, camera);
		returnValue = application.run();
	}

	catch (std::exception& e)
	{
		std::cout << std::endl << "Exception: " << std::endl  << std::endl << e.what() << std::endl << std::endl;
	}

	catch (...)
	{
		std::cout << std::endl << "Unknown error" << std::endl << std::endl;
	}

	freeGraphicsBuffers();

	if (g_hVerticesCopy != 0)
	{
		free(g_hVerticesCopy);
	}

	if (g_hEdgesCopy != 0)
	{
		free(g_hEdgesCopy);
	}

	if (g_hGraphCopy != 0)
	{
		free(g_hGraphCopy);
	}

	if (g_hPrimitives != 0)
	{
		free(g_hPrimitives);
	}

#ifdef USE_QUADTREE

	if (g_dQuadtree != 0)
	{
		FREE_ON_DEVICE(g_dQuadtree);
	}

	freeQuadtreeBuffers();
#endif

	if (g_dGraph != 0)
	{
		FREE_ON_DEVICE(g_dGraph);
	}

	if (g_dConfiguration != 0)
	{
		FREE_ON_DEVICE(g_dConfiguration);
	}

	if (g_hConfiguration != 0)
	{
		free(g_hConfiguration);
	}
	
	freeGraphBuffers();
	freeImageMaps();
	freeSamplingBuffers();
	freeWorkQueues();

	// DEBUG:
	system("pause");
	return returnValue;
}
