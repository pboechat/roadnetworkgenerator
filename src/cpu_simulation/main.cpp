// memory leak detection
//#include <vld.h>

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
	if (g_configuration != 0)
	{
		free(g_configuration);
	}

	g_configuration = (Configuration*)malloc(sizeof(Configuration));
	g_configuration->loadFromFile(configurationFile);
	allocateAndInitializeImageMaps(g_configuration->populationDensityMapFilePath,
								   g_configuration->waterBodiesMapFilePath,
								   g_configuration->blockadesMapFilePath,
								   g_configuration->naturalPatternMapFilePath,
								   g_configuration->radialPatternMapFilePath,
								   g_configuration->rasterPatternMapFilePath,
								   g_configuration->worldWidth,
								   g_configuration->worldHeight);
	allocateSamplingBuffers(g_configuration->samplingArc);
	allocateWorkQueues(g_configuration->maxWorkQueueCapacity);
	Box2D worldBounds(0.0f, 0.0f, (float)g_configuration->worldWidth, (float)g_configuration->worldHeight);
	renderer.setUpImageMaps(worldBounds, g_populationDensityMap, g_waterBodiesMap, g_blockadesMap);

	//////////////////////////////////////////////////////////////////////////
	//	ALLOCATE GRAPH
	//////////////////////////////////////////////////////////////////////////
	if (g_graph != 0)
	{
		free(g_graph);
	}

	if (g_graphCopy != 0)
	{
		free(g_graphCopy);
	}

	g_graph = (RoadNetworkGraph::Graph*)malloc(sizeof(RoadNetworkGraph::Graph));
	g_graphCopy = (RoadNetworkGraph::BaseGraph*)malloc(sizeof(RoadNetworkGraph::BaseGraph));

	allocateGraphBuffers(g_configuration->maxVertices, g_configuration->maxEdges);
#ifdef USE_QUADTREE
	//////////////////////////////////////////////////////////////////////////
	//	ALLOCATE QUADTREE
	//////////////////////////////////////////////////////////////////////////

	if (g_quadtree != 0)
	{
		free(g_quadtree);
	}

	g_quadtree = new RoadNetworkGraph::QuadTree();
	allocateQuadtreeBuffers(g_configuration->maxResultsPerQuery);
	RoadNetworkGraph::initializeQuadtree(g_quadtree, worldBounds, g_configuration->quadtreeDepth, g_configuration->maxResultsPerQuery, g_quadrants, g_quadrantsEdges);

	RoadNetworkGraph::initializeGraph(g_graph, g_configuration->snapRadius, g_configuration->maxVertices, g_configuration->maxEdges, g_vertices, g_edges, g_quadtree, g_configuration->maxResultsPerQuery, g_queryResults);
#else
	RoadNetworkGraph::initializeGraph(g_graph, g_configuration->snapRadius, g_configuration->maxVertices, g_configuration->maxEdges, g_vertices, g_edges);
#endif
	RoadNetworkGraph::initializeBaseGraph(g_graphCopy, g_verticesCopy, g_edgesCopy);

	allocatePrimitivesBuffer(g_configuration->maxPrimitives);

	RoadNetworkGenerator generator;

#ifdef _DEBUG
	Timer timer;
	timer.start();
#endif
	generator.execute();
#ifdef _DEBUG
	timer.end();
	std::cout << "*****************************" << std::endl;
	std::cout << "	DETAILS:				   " << std::endl;
	std::cout << "*****************************" << std::endl;
#ifdef _DEBUG
	std::cout << "seed: " << g_configuration->seed << std::endl;
#endif
	std::cout << "generation time: " << timer.elapsedTime() << " seconds" << std::endl;
	std::cout << "last highway derivation (max./real): " << g_configuration->maxHighwayDerivation << " / " << generator.getLastHighwayDerivation() << std::endl;
	std::cout << "last street derivation (max./real): " << g_configuration->maxStreetDerivation << " / " << generator.getLastStreetDerivation() << std::endl;
	std::cout << "work queue capacity (max/max. in use): " << g_configuration->maxWorkQueueCapacity << " / " << generator.getMaxWorkQueueCapacityUsed() << std::endl;
	std::cout << "memory (allocated/in use): " << toMegabytes(getAllocatedMemory(g_graph)) << " MB / " << toMegabytes(getMemoryInUse(g_graph)) << " MB" << std::endl;
	std::cout << "vertices (allocated/in use): " << getAllocatedVertices(g_graph) << " / " << getVerticesInUse(g_graph) << std::endl;
	std::cout << "edges (allocated/in use): " << getAllocatedEdges(g_graph) << " / " << getEdgesInUse(g_graph) << std::endl;
	std::cout << "vertex in connections (max./max. in use): " << getMaxVertexInConnections(g_graph) << " / " << getMaxVertexInConnectionsInUse(g_graph) << std::endl;
	std::cout << "vertex out connections (max./max. in use): " << getMaxVertexOutConnections(g_graph) << " / " << getMaxVertexOutConnectionsInUse(g_graph) << std::endl;
	std::cout << "avg. vertex in connections in use: " << getAverageVertexInConnectionsInUse(g_graph) << std::endl;
	std::cout << "avg. vertex out connections in use: " << getAverageVertexOutConnectionsInUse(g_graph) << std::endl;
#ifdef USE_QUADTREE
	std::cout << "edges per quadrant (max./max. in use): " << getMaxEdgesPerQuadrant(g_graph) << " / " << getMaxEdgesPerQuadrantInUse(g_graph) << std::endl;
#endif
	std::cout << "num. collision checks: " << getNumCollisionChecks(g_graph) << std::endl;
	std::cout  << std::endl << std::endl;
#endif

	geometry.build();
	labels.build();
	camera.centerOnTarget(worldBounds);
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

		Camera camera(screenWidth, screenHeight, FOVY_DEG, ZNEAR, ZFAR);
		RoadNetworkGeometry geometry;
		RoadNetworkLabels labels;
		SceneRenderer renderer(camera, geometry, labels);
		RoadNetworkInputController inputController(camera, configurationFile, renderer, geometry, labels, generateAndDisplay);
		application.setCamera(camera);
		application.setRenderer(renderer);
		application.setInputController(inputController);
		initializeWorkQueues();
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

	if (g_graphCopy != 0)
	{
		delete g_graphCopy;
	}

#ifdef USE_QUADTREE

	if (g_quadtree != 0)
	{
		delete g_quadtree;
	}

	freeQuadtreeBuffers();
#endif

	if (g_graph != 0)
	{
		delete g_graph;
	}

	if (g_configuration != 0)
	{
		delete g_configuration;
	}

	freePrimitivesBuffer();
	freeGraphBuffers();
	freeImageMaps();
	freeSamplingBuffers();
	freeWorkQueues();
	// DEBUG:
	system("pause");
	return returnValue;
}
