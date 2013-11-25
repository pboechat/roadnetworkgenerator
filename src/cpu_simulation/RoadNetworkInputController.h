#ifndef ROADNETWORKINPUTCONTROLLER_H
#define ROADNETWORKINPUTCONTROLLER_H

#include <InputController.h>
#include <SceneRenderer.h>
#include <Configuration.h>
#include <Camera.h>
#include <Graph.h>
#include <RoadNetworkGenerator.h>
#include <RoadNetworkGeometry.h>
#include <AABB.h>

#include <string>
#include <iostream>

class RoadNetworkInputController : public InputController
{
public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	RoadNetworkInputController(Camera& camera, 
							   const std::string& configurationFile,
							   SceneRenderer& sceneRenderer,
							   RoadNetworkGeometry& roadNetworkGeometry) 
		: 
		InputController(camera, 20.0f, 10.0f),
		configurationFile(configurationFile),
		sceneRenderer(sceneRenderer),
		roadNetworkGeometry(roadNetworkGeometry)
	{
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	virtual void update(double deltaTime)
	{
		if (getKey(VK_ESCAPE))
		{
			Application::instance->halt();
		}

		if (getKey(VK_LEFT) || getKey(65))
		{
			moveCameraLeft((float)deltaTime);
		}

		else if (getKey(VK_RIGHT) || getKey(68))
		{
			moveCameraRight((float)deltaTime);
		}

		if (getKey(VK_UP) || getKey(87))
		{
			moveCameraUp((float)deltaTime);
		}

		else if (getKey(VK_DOWN) || getKey(83))
		{
			moveCameraDown((float)deltaTime);
		}

		if (getKey(81) || getKey(33))
		{
			moveCameraForward((float)deltaTime);
		}

		else if (getKey(69) || getKey(34))
		{
			moveCameraBackward((float)deltaTime);
		}

		if (getKey(VK_F5))
		{
			// reload configuration
			Configuration configuration;
			configuration.loadFromFile(configurationFile);

			AABB worldBounds(0.0f, 0.0f, (float)configuration.worldWidth, (float)configuration.worldHeight);
			RoadNetworkGraph::Graph roadNetwork(worldBounds, (float)configuration.quadtreeCellArea, (float)configuration.quadtreeQueryRadius);

			// regenerate road network graph
			RoadNetworkGenerator roadNetworkGenerator;
			roadNetworkGenerator.execute(configuration, roadNetwork);

			// rebuild road network geometry
			roadNetworkGeometry.build(roadNetwork, configuration.highwayColor, configuration.streetColor);

			sceneRenderer.setWorldBounds(worldBounds);

			camera.centerOnTarget(worldBounds);
		}

		if (getLeftMouseButtonDown())
		{
			glm::vec2 mousePosition = getMousePosition();
			std::cout << "(" << mousePosition.x << ", " << (camera.getScreenHeight() - mousePosition.y) << ")" << std::endl;
		}
	}

private:
	std::string configurationFile;
	SceneRenderer& sceneRenderer;
	RoadNetworkGeometry& roadNetworkGeometry;

};

#endif