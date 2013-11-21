// memory leak detection
//#include <vld.h>

#include <Application.h>
#include <Camera.h>
#include <RoadNetworkInputController.h>
#include <SceneRenderer.h>
#include <RoadNetworkGeometry.h>
#include <Configuration.h>
#include <RoadNetworkGenerator.h>
#include <RoadNetwork.h>

#include <glm/glm.hpp>

#include <string>
#include <iostream>
#include <io.h>
#include <iomanip>
#include <random>

#define ZNEAR 10.0f
#define ZFAR 10000.0f
#define FOVY_DEG 60.0f
#define HALF_PI 1.570796325f

//////////////////////////////////////////////////////////////////////////
void printUsage()
{
	std::cerr << "Command line options: <configuration file>";
}

//////////////////////////////////////////////////////////////////////////
void centerWorldOnScreen(const Configuration& configuration, Camera& camera)
{
	float screenDiagonal = glm::sqrt(glm::pow((float)configuration.worldWidth, 2.0f) + glm::pow((float)configuration.worldHeight, 2.0f));
	float distance = (configuration.worldWidth / 2.0f) / glm::tan(glm::radians(camera.getFovY() / 2.0f));
	camera.localTransform.position = glm::vec3(configuration.worldWidth / 2.0f, configuration.worldHeight / 2.0f, distance);
}

//////////////////////////////////////////////////////////////////////////
void centerGeometryOnScreen(const RoadNetworkGeometry& geometry, Camera& camera)
{
	float screenDiagonal = glm::sqrt(glm::pow(geometry.bounds.getExtents().x, 2.0f) + glm::pow(geometry.bounds.getExtents().y, 2.0f));
	float distance = (screenDiagonal / 2.0f) / glm::tan(glm::radians(camera.getFovY() / 2.0f));
	camera.localTransform.position = glm::vec3(geometry.bounds.min.x + geometry.bounds.getExtents().x / 2.0f, geometry.bounds.min.y + geometry.bounds.getExtents().y / 2.0f, distance);
}

//////////////////////////////////////////////////////////////////////////
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

		unsigned int seed;
		if (configuration.seed >= 0)
		{
			seed = configuration.seed;
		}
		else
		{
			seed = (unsigned int)time(0);
			std::cout << "seed: " << seed << std::endl;
		}
		srand(seed);

		Application application("Road Network Generator (CPU)", configuration.worldWidth, configuration.worldHeight);

		if (gl3wInit())
		{
			throw std::runtime_error("gl3wInit() failed");
		}

		Camera camera(configuration.worldWidth, configuration.worldHeight, FOVY_DEG, ZNEAR, ZFAR);
		RoadNetworkInputController inputController(camera);
		RoadNetworkGeometry roadNetworkGeometry;
		SceneRenderer renderer(configuration, camera, roadNetworkGeometry);

		application.setCamera(camera);
		application.setRenderer(renderer);
		application.setInputController(inputController);

		RoadNetwork::Graph roadNetwork(configuration);

		RoadNetworkGenerator roadNetworkGenerator;
		roadNetworkGenerator.execute(configuration, roadNetwork);

		roadNetworkGeometry.build(configuration, roadNetwork);

		centerWorldOnScreen(configuration, camera);
		//centerGeometryOnScreen(geometry, camera);

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
