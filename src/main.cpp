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
	allocateAndInitializeConfiguration(configurationFile);
	Box2D worldBounds(0.0f, 0.0f, (float)g_hConfiguration->worldWidth, (float)g_hConfiguration->worldHeight);
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
	allocateGraph(g_hConfiguration->maxVertices, g_hConfiguration->maxEdges);
#ifdef USE_QUADTREE
	allocateQuadtree(g_hConfiguration->maxResultsPerQuery, g_hConfiguration->maxQuadrants);
	INVOKE_GLOBAL_CODE7(RoadNetworkGraph::initializeQuadtree, 1, 1, g_dQuadtree, worldBounds, g_hConfiguration->quadtreeDepth, g_hConfiguration->maxResultsPerQuery, g_hConfiguration->maxQuadrants, g_dQuadrants, g_dQuadrantsEdges);
#endif
#ifdef USE_QUADTREE
	INVOKE_GLOBAL_CODE9(RoadNetworkGraph::initializeGraph, 1, 1, g_dGraph, g_hConfiguration->snapRadius, g_hConfiguration->maxVertices, g_hConfiguration->maxEdges, g_dVertices, g_dEdges, g_dQuadtree, g_hConfiguration->maxResultsPerQuery, g_dQueryResults);
#else
	INVOKE_GLOBAL_CODE6(RoadNetworkGraph::initializeGraph, 1, 1, g_dGraph, g_hConfiguration->snapRadius, g_hConfiguration->maxVertices, g_hConfiguration->maxEdges, g_dVertices, g_dEdges);
#endif
	allocatePrimitives();

	RoadNetworkGenerator generator;

#ifdef _DEBUG
	Timer timer;
	timer.start();
#endif
	generator.execute();
#ifdef _DEBUG
	timer.end();

#ifdef USE_QUADTREE
	copyQuadtreeToHost();
#endif

	std::cout << "*****************************" << std::endl;
	std::cout << "	DETAILS:				   " << std::endl;
	std::cout << "*****************************" << std::endl;
#ifdef _DEBUG
	std::cout << "seed: " << g_hConfiguration->seed << std::endl;
#endif
	std::cout << "generation time: " << timer.elapsedTime() << " seconds" << std::endl;
	std::cout << "last highway derivation (max./real): " << g_hConfiguration->maxHighwayDerivation << " / " << generator.getLastHighwayDerivation() << std::endl;
	std::cout << "last street derivation (max./real): " << g_hConfiguration->maxStreetDerivation << " / " << generator.getLastStreetDerivation() << std::endl;
	std::cout << "work queue capacity (max/max. in use): " << (g_hConfiguration->maxWorkQueueCapacity * NUM_PROCEDURES) << " / " << generator.getMaxWorkQueueCapacityUsed() << std::endl;
	std::cout << "memory (allocated/in use): " << toMegabytes(getAllocatedMemory(g_hGraph)) << " MB / " << toMegabytes(getMemoryInUse(g_hGraph)) << " MB" << std::endl;
	std::cout << "vertices (allocated/in use): " << getAllocatedVertices(g_hGraph) << " / " << getVerticesInUse(g_hGraph) << std::endl;
	std::cout << "edges (allocated/in use): " << getAllocatedEdges(g_hGraph) << " / " << getEdgesInUse(g_hGraph) << std::endl;
	std::cout << "vertex in connections (max./max. in use): " << getMaxVertexInConnections(g_hGraph) << " / " << getMaxVertexInConnectionsInUse(g_hGraph) << std::endl;
	std::cout << "vertex out connections (max./max. in use): " << getMaxVertexOutConnections(g_hGraph) << " / " << getMaxVertexOutConnectionsInUse(g_hGraph) << std::endl;
	std::cout << "avg. vertex in connections in use: " << getAverageVertexInConnectionsInUse(g_hGraph) << std::endl;
	std::cout << "avg. vertex out connections in use: " << getAverageVertexOutConnectionsInUse(g_hGraph) << std::endl;
#ifdef USE_QUADTREE
	std::cout << "edges per quadrant (max./max. in use): " << MAX_EDGES_PER_QUADRANT << " / " << getMaxEdgesPerQuadrantInUse(g_hQuadtree) << std::endl;
	std::cout << "results per query (max./max. in use): " << g_hConfiguration->maxResultsPerQuery << " / " << getMaxResultsPerQueryInUse(g_hQuadtree) << std::endl;
#endif
	std::cout << "primitive size (max./max. in use): " << MAX_VERTICES_PER_PRIMITIVE << "/" << generator.getMaxPrimitiveSize() << std::endl;
	std::cout << "num. collision checks: " << getNumCollisionChecks(g_hGraph) << std::endl;
	std::cout  << std::endl << std::endl;
#endif

	allocateGraphicsBuffers(g_hConfiguration->vertexBufferSize, g_hConfiguration->indexBufferSize);

	renderer.setUpImageMaps(worldBounds, g_hPopulationDensityMap, g_hWaterBodiesMap, g_hBlockadesMap);

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
	freePrimitives();
#ifdef USE_QUADTREE
	freeQuadtree();
#endif
	freeGraph();
	freeImageMaps();
	freeSamplingBuffers();
	freeWorkQueues();
	freeConfiguration();

	// DEBUG:
	system("pause");
	return returnValue;
}
