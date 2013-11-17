#ifndef INPUTCONTROLLER_H
#define INPUTCONTROLLER_H

#include <Camera.h>

#include <Windows.h>

#define CAMERA_PITCH_LIMIT 45.0f
#define NUMBER_OF_VIRTUAL_KEYS 0xFF

class InputController
{
public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	virtual void update(double deltaTime) = 0;

	////////////////////////////////////////////////////////////////////////////////////////////////////
	void swapBuffers()
	{
		back = front;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	void keyDown(unsigned int virtualKey)
	{
		front.keys[virtualKey] = true;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	void keyUp(unsigned int virtualKey)
	{
		front.keys[virtualKey] = false;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	void mouseButtonDown(unsigned int button, int x, int y)
	{
		if (button == MK_RBUTTON)
		{
			front.rightMouseButton = true;
		}

		else if (button == MK_LBUTTON)
		{
			front.leftMouseButton = true;
		}
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	void mouseButtonUp(unsigned int button, int x, int y)
	{
		if (button == MK_RBUTTON)
		{
			front.rightMouseButton = false;
		}

		else if (button == MK_LBUTTON)
		{
			front.leftMouseButton = false;
		}
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	virtual void mouseMove(int x, int y)
	{
		front.mousePosition = glm::vec2((float)x, (float)y);
	}

protected:
	Camera& camera;

	////////////////////////////////////////////////////////////////////////////////////////////////////
	InputController(Camera& camera, float moveSpeed, float rotationSpeed) : camera(camera), moveSpeed(moveSpeed), rotationSpeed(rotationSpeed), cameraYaw(0), cameraPitch(0), cameraRoll(0)
	{
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	void defaultNavigation(double deltaTime)
	{
		if (getKey(VK_LEFT) || getKey(65))
		{
			moveCameraLeft((float)deltaTime);
		}

		else if (getKey(VK_RIGHT) || getKey(68))
		{
			moveCameraRight((float)deltaTime);
		}

		else if (getKey(VK_UP) || getKey(87))
		{
			moveCameraForward((float)deltaTime);
		}

		else if (getKey(VK_DOWN) || getKey(83))
		{
			moveCameraBackward((float)deltaTime);
		}

		else if (getKey(81) || getKey(33))
		{
			moveCameraUp((float)deltaTime);
		}

		else if (getKey(69) || getKey(34))
		{
			moveCameraDown((float)deltaTime);
		}

		if (!getRightMouseButton())
		{
			return;
		}

		glm::vec2 mousePosition = getMousePosition();

		if (lastMousePosition.x == mousePosition.x && lastMousePosition.y == mousePosition.y)
		{
			return;
		}

		glm::vec2 mouseDirection = glm::normalize(mousePosition - lastMousePosition);

		if (mouseDirection.x > 0)
		{
			turnCameraRight((float)deltaTime);
		}

		else if (mouseDirection.x < 0)
		{
			turnCameraLeft((float)deltaTime);
		}

		if (mouseDirection.y > 0)
		{
			turnCameraUp((float)deltaTime);
		}

		else if (mouseDirection.y < 0)
		{
			turnCameraDown((float)deltaTime);
		}

		lastMousePosition = mousePosition;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline bool getKey(unsigned int keyCode) const
	{
		return back.keys[keyCode];
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline bool getRightMouseButton() const
	{
		return back.rightMouseButton;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline bool getLefttMouseButton() const
	{
		return back.leftMouseButton;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline glm::vec2 getMousePosition() const
	{
		return back.mousePosition;
	}

private:
	struct InputBuffer
	{
		bool keys[NUMBER_OF_VIRTUAL_KEYS];
		bool rightMouseButton;
		bool leftMouseButton;
		glm::vec2 mousePosition;

		InputBuffer() : rightMouseButton(false), leftMouseButton(false)
		{
			memset(keys, 0, sizeof(bool) * NUMBER_OF_VIRTUAL_KEYS);
		}

		InputBuffer& operator = (InputBuffer& other)
		{
			memcpy(keys, other.keys, sizeof(bool) * NUMBER_OF_VIRTUAL_KEYS);
			rightMouseButton = other.rightMouseButton;
			leftMouseButton = other.leftMouseButton;
			mousePosition = other.mousePosition;
			return *this;
		}

	};

	InputBuffer front;
	InputBuffer back;
	float moveSpeed;
	float rotationSpeed;
	glm::vec2 lastMousePosition;
	float cameraYaw;
	float cameraPitch;
	float cameraRoll;

	////////////////////////////////////////////////////////////////////////////////////////////////////
	void moveCameraLeft(float deltaTime)
	{
		camera.localTransform.position -= camera.localTransform.right() * moveSpeed * deltaTime;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	void moveCameraRight(float deltaTime)
	{
		camera.localTransform.position += camera.localTransform.right() * moveSpeed * deltaTime;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	void moveCameraForward(float deltaTime)
	{
		camera.localTransform.position += camera.localTransform.forward() * moveSpeed * deltaTime;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	void moveCameraBackward(float deltaTime)
	{
		camera.localTransform.position -= camera.localTransform.forward() * moveSpeed * deltaTime;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	void moveCameraUp(float deltaTime)
	{
		camera.localTransform.position += glm::vec3(0, 1, 0) * moveSpeed * deltaTime;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	void moveCameraDown(float deltaTime)
	{
		camera.localTransform.position -= glm::vec3(0, 1, 0) * moveSpeed * deltaTime;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	void turnCameraUp(float deltaTime)
	{
		cameraPitch = glm::clamp(cameraPitch - rotationSpeed * deltaTime, -CAMERA_PITCH_LIMIT, CAMERA_PITCH_LIMIT);
		updateCameraRotation();
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	void turnCameraDown(float deltaTime)
	{
		cameraPitch = glm::clamp(cameraPitch + rotationSpeed * deltaTime, -CAMERA_PITCH_LIMIT, CAMERA_PITCH_LIMIT);
		updateCameraRotation();
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	void turnCameraLeft(float deltaTime)
	{
		cameraYaw += rotationSpeed * deltaTime;
		updateCameraRotation();
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	void turnCameraRight(float deltaTime)
	{
		cameraYaw -= rotationSpeed * deltaTime;
		updateCameraRotation();
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	void updateCameraRotation()
	{
		glm::vec3 forward = glm::angleAxis(cameraPitch, glm::vec3(1, 0, 0)) * glm::angleAxis(cameraYaw, glm::vec3(0, 1, 0)) * glm::vec3(0, 0, -1);
		camera.localTransform.lookAt(camera.localTransform.position + forward);
	}

};


#endif