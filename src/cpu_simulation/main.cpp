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

#include <glm/glm.hpp>

#include <string>
#include <iostream>
#include <io.h>
#include <iomanip>

#define ZNEAR 10.0f
#define ZFAR 10000.0f
#define FOVY_DEG 60.0f
#define HALF_PI 1.570796325f

void printUsage()
{
	std::cerr << "Command line options: <configuration file>";
}

int main(int argc, char** argv)
{
	try
	{
		if (argc < 2)
		{
			printUsage();
			exit(EXIT_FAILURE);
		}

		std::string configurationFile = argv[1];

		if (configurationFile.empty())
		{
			printUsage();
			exit(EXIT_FAILURE);
		}

		Configuration configuration;
		configuration.loadFromFile(configurationFile);

		// TODO:
		unsigned int screenWidth = configuration.worldWidth;
		unsigned int screenHeight = configuration.worldHeight;

		Application application("Road Network Generator (CPU)", screenWidth, screenHeight);

		if (gl3wInit())
		{
			throw std::runtime_error("gl3wInit() failed");
		}

		RoadNetworkGeometry roadNetworkGeometry;

		Camera camera(screenWidth, screenHeight, FOVY_DEG, ZNEAR, ZFAR);
		RoadNetworkInputController inputController(camera, configurationFile, roadNetworkGeometry);
		SceneRenderer renderer(camera, screenWidth, screenHeight, roadNetworkGeometry, configuration.populationDensityMap, configuration.waterBodiesMap);

		application.setCamera(camera);
		application.setRenderer(renderer);
		application.setInputController(inputController);

		AABB worldBounds(0.0f, 0.0f, (float)configuration.worldWidth, (float)configuration.worldHeight);
		RoadNetworkGraph::Graph roadNetwork(worldBounds, (float)configuration.quadtreeCellArea, (float)configuration.quadtreeQueryRadius);

		RoadNetworkGenerator roadNetworkGenerator;
		roadNetworkGenerator.execute(configuration, roadNetwork);

		roadNetworkGeometry.build(roadNetwork, configuration.highwayColor, configuration.streetColor);

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
