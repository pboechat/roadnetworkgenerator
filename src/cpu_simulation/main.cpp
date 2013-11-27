// memory leak detection
//#include <vld.h>

#include <Application.h>
#include <Camera.h>
#include <RoadNetworkInputController.h>
#include <SceneRenderer.h>
#include <RoadNetworkGeometry.h>
#include <Configuration.h>
#include <RoadNetworkGenerator.h>
#include <AABB.h>
#include <Graph.h>
#include <Timer.h>

#include <glm/glm.hpp>

#include <string>
#include <iostream>
#include <io.h>
#include <iomanip>

#define ZNEAR 10.0f
#define ZFAR 10000.0f
#define FOVY_DEG 60.0f
#define HALF_PI 1.570796325f

#ifdef _DEBUG
#define toKilobytes(a) (a / 1024)
#define toMegabytes(a) (a / 1048576)
#endif

void printUsage()
{
	std::cerr << "Command line options: <width> <height> <configuration file>";
}

int main(int argc, char** argv)
{
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

		Configuration configuration;
		configuration.loadFromFile(configurationFile);

		Application application("Road Network Generator (CPU)", screenWidth, screenHeight);

		if (gl3wInit())
		{
			throw std::runtime_error("gl3wInit() failed");
		}

		RoadNetworkGeometry geometry;

		Camera camera(screenWidth, screenHeight, FOVY_DEG, ZNEAR, ZFAR);
		SceneRenderer renderer(camera, geometry, configuration.populationDensityMap, configuration.waterBodiesMap);
		RoadNetworkInputController inputController(camera, configurationFile, renderer, geometry);

		application.setCamera(camera);
		application.setRenderer(renderer);
		application.setInputController(inputController);

		AABB worldBounds(0.0f, 0.0f, (float)configuration.worldWidth, (float)configuration.worldHeight);
#ifdef USE_QUADTREE
		RoadNetworkGraph::Graph graph(worldBounds, configuration.quadtreeDepth, configuration.snapRadius, configuration.maxVertices, configuration.maxEdges, configuration.maxResultsPerQuery);
#else
		RoadNetworkGraph::Graph graph(worldBounds, configuration.snapRadius, configuration.maxVertices, configuration.maxEdges, configuration.maxResultsPerQuery);
#endif
		RoadNetworkGenerator generator;
#ifdef _DEBUG
		Timer timer;
		timer.start();
#endif
		generator.execute(configuration, graph);
#ifdef _DEBUG
		timer.end();
		std::cout << "*****************************" << std::endl;
		std::cout << "	STATISTICS:				   " << std::endl;
		std::cout << "*****************************" << std::endl;
		std::cout << "generation time: " << timer.elapsedTime() << " seconds" << std::endl;
		std::cout << "memory (allocated/in use): " << toMegabytes(graph.getAllocatedMemory()) << " MB / " << toMegabytes(graph.getMemoryInUse()) << " MB" << std::endl;
		std::cout << "vertices (allocated/in use): " << graph.getAllocatedVertices() << " / " << graph.getVerticesInUse() << std::endl;
		std::cout << "edges (allocated/in use): " << graph.getAllocatedEdges() << " / " << graph.getEdgesInUse() << std::endl;
		std::cout << "vertex in connections (max./max. in use): " << graph.getMaxVertexInConnections() << " / " << graph.getMaxVertexInConnectionsInUse() << std::endl;
		std::cout << "vertex out connections (max./max. in use): " << graph.getMaxVertexOutConnections() << " / " << graph.getMaxVertexOutConnectionsInUse() << std::endl;
		std::cout << "avg. vertex in connections in use: " << graph.getAverageVertexInConnectionsInUse() << std::endl;
		std::cout << "avg. vertex out connections in use: " << graph.getAverageVertexOutConnectionsInUse() << std::endl;
#ifdef USE_QUADTREE
		std::cout << "edges per quadrant (max./max. in use): " << graph.getMaxEdgesPerQuadrant() << " / " << graph.getMaxEdgesPerQuadrantInUse() << std::endl;
#endif
		std::cout << "num. collision checks: " << graph.getNumCollisionChecks() << std::endl;
		std::cout  << std::endl << std::endl;
#endif

		geometry.build(graph, configuration.highwayColor, configuration.streetColor);
		renderer.setWorldBounds(worldBounds);
		camera.centerOnTarget(worldBounds);

		return application.run();
	}

	catch (std::exception& e)
	{
		std::cout << "Exception: " << std::endl  << std::endl << e.what() << std::endl << std::endl;
	}

	catch (...)
	{
		std::cout << "Unknown error" << std::endl << std::endl;
	}

	// DEBUG:
	system("pause");
	return -1;
}
