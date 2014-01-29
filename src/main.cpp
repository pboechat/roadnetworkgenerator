// memory leak detection
//#include <vld.h>

#include "Defines.h"
#include <RoadNetworkInputController.h>
#include <SceneRenderer.h>
#include <RoadNetworkGraphGenerator.h>
#include <RoadNetworkGeometryGenerator.h>
#include <RoadNetworkLabelsGenerator.h>
#include <Configuration.cuh>
#include <ImageMap.cuh>

#include <Application.h>
#include <Camera.h>
#include <Box2D.cuh>
#include <Timer.h>
#include <MathExtras.cuh>

#include <vector_math.h>

#include <string>
#include <iostream>
#include <io.h>
#include <iomanip>

void printUsage()
{
	std::cerr << "Command line options: <width> <height> <configuration file>";
}

void generateAndDisplay(const std::string& configurationFile, SceneRenderer& renderer, RoadNetworkGeometryGenerator& geometryGenerator, RoadNetworkLabelsGenerator& labelsGenerator, Camera& camera)
{
	Configuration configuration;
	configuration.loadFromFile(configurationFile);

	renderer.readConfigurations(configuration);
	geometryGenerator.readConfigurations(configuration);

	ImageMap populationDensityMap, waterBodiesMap, blockadesMap, naturalPatternMap, radialPatternMap, rasterPatternMap;

	RoadNetworkGraphGenerator graphGenerator(configuration, populationDensityMap, waterBodiesMap, blockadesMap, naturalPatternMap, radialPatternMap, rasterPatternMap);
	graphGenerator.addObserver(&geometryGenerator);
	graphGenerator.addObserver(&labelsGenerator);
	graphGenerator.execute();

	Box2D worldBounds(0.0f, 0.0f, (float)configuration.worldWidth, (float)configuration.worldHeight);
	renderer.setUpImageMaps(worldBounds, populationDensityMap, waterBodiesMap, blockadesMap);

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
		RoadNetworkGeometryGenerator geometry;
		RoadNetworkLabelsGenerator labels;
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

	// DEBUG:
	system("pause");
	return returnValue;
}
