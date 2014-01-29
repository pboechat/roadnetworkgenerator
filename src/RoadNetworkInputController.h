#ifndef ROADNETWORKINPUTCONTROLLER_H
#define ROADNETWORKINPUTCONTROLLER_H

#include <Application.h>
#include <InputController.h>
#include <SceneRenderer.h>
#include <Camera.h>
#include <RoadNetworkGraphGenerator.h>
#include <RoadNetworkGeometryGenerator.h>
#include <RoadNetworkLabelsGenerator.h>

#include <string>

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

		if (getKeyDown(VK_F1))
		{
			renderer.togglePopulationDensityMap();
		}

		if (getKeyDown(VK_F2))
		{
			renderer.toggleWaterBodiesMap();
		}

		if (getKeyDown(VK_F3))
		{
			renderer.toggleBlockadesMap();
		}

		if (getKeyDown(VK_F5))
		{
			callback(configurationFile, renderer, geometryGenerator, labelsGenerator, camera);
		}
	}

private:
	std::string configurationFile;
	SceneRenderer& renderer;
	RoadNetworkGeometryGenerator& geometryGenerator;
	RoadNetworkLabelsGenerator& labelsGenerator;
	GenerateAndDisplayCallback callback;

};

#endif