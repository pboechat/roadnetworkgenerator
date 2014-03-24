#ifndef ROADNETWORKINPUTCONTROLLER_H
#define ROADNETWORKINPUTCONTROLLER_H

#pragma once

#include <Application.h>
#include <InputController.h>
#include <SceneRenderer.h>
#include <Camera.h>
#include <RoadNetworkGraphGenerator.h>
#include <RoadNetworkGeometryGenerator.h>
#include <RoadNetworkLabelsGenerator.h>

#include <string>
#include <iostream>

class RoadNetworkInputController : public InputController
{
public:
	typedef void (*GenerateAndDisplayCallback)(const std::string&, SceneRenderer&, RoadNetworkGeometryGenerator&, RoadNetworkLabelsGenerator& labels, Camera&);

	RoadNetworkInputController(Camera& camera,
							   const std::string& configurationFile,
							   SceneRenderer& renderer,
							   RoadNetworkGeometryGenerator& geometryGenerator,
							   RoadNetworkLabelsGenerator& labelsGenerator,
							   GenerateAndDisplayCallback callback)
		:
		InputController(camera, 100.0f, 10.0f),
		configurationFile(configurationFile),
		renderer(renderer),
		geometryGenerator(geometryGenerator),
		labelsGenerator(labelsGenerator),
		callback(callback)
	{
	}

	//////////////////////////////////////////////////////////////////////////
	virtual void update(double deltaTime)
	{
		defaultNavigation(deltaTime);

		if (getKey(VK_ESCAPE))
		{
			Application::instance->halt();
		}

		if (getKeyDown(VK_F1))
		{
			renderer.togglePopulationDensityShadingType();
		}

		if (getKeyDown(VK_F2))
		{
			renderer.toggleWaterBodiesMap();
		}

		if (getKeyDown(VK_F3))
		{
			renderer.toggleBlockadesMap();
		}

		if (getKeyDown(VK_F4))
		{
			renderer.toggleDrawQuadtree();
		}

		if (getKeyDown(VK_F5))
		{
			callback(configurationFile, renderer, geometryGenerator, labelsGenerator, camera);
		}

		// DEBUG:
		if (getKeyDown(VK_F6))
		{
			printCameraStats();
		}
	}

	//////////////////////////////////////////////////////////////////////////
	inline void printCameraStats()
	{
		std::cout << "Position: (" << camera.localTransform.position.x << ", " << camera.localTransform.position.y << ", " << camera.localTransform.position.z << ")" << std::endl
				  << "Rotation: (" << camera.localTransform.rotation.x << ", " << camera.localTransform.rotation.y << ", " << camera.localTransform.rotation.z << ", " << camera.localTransform.rotation.w << ")" << std::endl;
	}

private:
	std::string configurationFile;
	SceneRenderer& renderer;
	RoadNetworkGeometryGenerator& geometryGenerator;
	RoadNetworkLabelsGenerator& labelsGenerator;
	GenerateAndDisplayCallback callback;

};

#endif