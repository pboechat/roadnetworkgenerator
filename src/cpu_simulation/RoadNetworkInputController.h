#ifndef ROADNETWORKINPUTCONTROLLER_H
#define ROADNETWORKINPUTCONTROLLER_H

#include <InputController.h>
#include <Camera.h>

class RoadNetworkInputController : public InputController
{
public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	RoadNetworkInputController(Camera& camera) : InputController(camera, 10.0f, 10.0f)
	{
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	virtual void update(double deltaTime)
	{
		InputController::defaultNavigation(deltaTime);

		if (getKey(VK_ESCAPE))
		{
			Application::instance->halt();
		}
	}

};

#endif