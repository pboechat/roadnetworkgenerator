// memory leak detection
#include <vld.h>

#include <Application.h>
#include <Camera.h>
#include <RoadNetworkInputController.h>
#include <RoadNetworkRenderer.h>
#include <RoadNetworkGeometry.h>
#include <Configuration.h>
#include <RoadNetworkGenerator.h>
#include <ImageMap.h>

#include <glm/glm.hpp>

#include <string>
#include <iostream>
#include <io.h>
#include <iomanip>

#define DEFAULT_SCREEN_WIDTH 1024
#define DEFAULT_SCREEN_HEIGHT 768
#define ZNEAR 0.3f
#define ZFAR 1000.0f
#define FOVY_DEG 60.0f
#define HALF_PI 1.570796325f

//////////////////////////////////////////////////////////////////////////
void printUsage()
{
	std::cerr << "Command line options: <configuration file>";
}

//////////////////////////////////////////////////////////////////////////
void centerGeometryOnScreen(RoadNetworkGeometry& geometry, Camera& camera)
{
	float width = geometry.bounds.extents().x;
	float height = geometry.bounds.extents().y;
	float screenDiagonal = glm::sqrt(glm::pow(width, 2.0f) + glm::pow(height, 2.0f) + 1.0f);
	float distance = glm::min((screenDiagonal / 2.0f) / glm::tan(glm::radians(camera.getFovY() / 2.0f)), camera.getFar());
	camera.localTransform.position = glm::vec3(geometry.bounds.min.x + width / 2.0f, geometry.bounds.min.y + height / 2.0f, distance);
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
		Application application("Road Network Generator (CPU)", DEFAULT_SCREEN_WIDTH, DEFAULT_SCREEN_HEIGHT);

		if (gl3wInit())
		{
			throw std::runtime_error("gl3wInit() failed");
		}

		RoadNetworkGeometry geometry;
		Camera camera(DEFAULT_SCREEN_WIDTH, DEFAULT_SCREEN_HEIGHT, FOVY_DEG, ZNEAR, ZFAR);
		RoadNetworkInputController inputController(camera);
		RoadNetworkRenderer renderer(camera, geometry);
		application.setCamera(camera);
		application.setRenderer(renderer);
		application.setInputController(inputController);
		// ---
		RoadNetworkGenerator roadNetworkGenerator;
		roadNetworkGenerator.execute(configuration, geometry);
		centerGeometryOnScreen(geometry, camera);
		// ---
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
