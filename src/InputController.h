#ifndef INPUTCONTROLLER_H
#define INPUTCONTROLLER_H

#pragma once

#include <Camera.h>
#include <VectorMath.h>

#include <Windows.h>

#define CAMERA_PITCH_LIMIT 45.0f
#define NUMBER_OF_VIRTUAL_KEYS 0xFF
#define MOUSE_SENSITIVITY 0.01f

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
	void afterUpdate()
	{
		clearTemporaryRegisters();
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	void keyDown(unsigned int virtualKey)
	{
		front.keys[virtualKey] = true;
		keysDown[virtualKey] = true;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	void keyUp(unsigned int virtualKey)
	{
		front.keys[virtualKey] = false;
		keysUp[virtualKey] = true;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	void mouseButtonDown(unsigned int button, int x, int y)
	{
		if (button == MK_RBUTTON)
		{
			front.rightMouseButton = true;
			rightMouseButtonDown = true;
		}

		else if (button == MK_LBUTTON)
		{
			front.leftMouseButton = true;
			leftMouseButtonDown = true;
		}
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	void mouseButtonUp(unsigned int button, int x, int y)
	{
		if (button == MK_RBUTTON)
		{
			front.rightMouseButton = false;
			rightMouseButtonUp = true;
		}

		else if (button == MK_LBUTTON)
		{
			front.leftMouseButton = false;
			leftMouseButtonUp = true;
		}
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	virtual void mouseMove(int x, int y)
	{
		front.mousePosition = vml_vec2((float)x, (float)y);
	}

protected:
	Camera& camera;
	vml_quat yaw;
	vml_quat pitch;

	////////////////////////////////////////////////////////////////////////////////////////////////////
	InputController(Camera& camera, float moveSpeed, float rotationSpeed) : camera(camera), moveSpeed(moveSpeed), rotationSpeed(rotationSpeed), rightMouseButtonUp(false), rightMouseButtonDown(false), leftMouseButtonUp(false), leftMouseButtonDown(false)
	{
		clearTemporaryRegisters();
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

		if (getKey(VK_UP) || getKey(87))
		{
			moveCameraForward((float)deltaTime);
		}

		else if (getKey(VK_DOWN) || getKey(83))
		{
			moveCameraBackward((float)deltaTime);
		}

		if (getKey(81) || getKey(33))
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

		vml_vec2 mousePosition = getMousePosition();

		glm::vec2 direction = mousePosition - lastMousePosition;
		vml_quat yawIncrement, pitchIncrement;

		if (direction.x >= MOUSE_SENSITIVITY)
		{
			yawIncrement = vml_angle_axis(-rotationSpeed * (float)deltaTime, vml_vec3(0.0f, 1.0f, 0.0f));
		}
		else if (direction.x <= -MOUSE_SENSITIVITY)
		{
			yawIncrement = vml_angle_axis(rotationSpeed * (float)deltaTime, vml_vec3(0.0f, 1.0f, 0.0f));
		}

		if (direction.y >= MOUSE_SENSITIVITY)
		{
			pitchIncrement = vml_angle_axis(-rotationSpeed * (float)deltaTime, vml_vec3(1.0f, 0.0f, 0.0f));
		}
		else if (direction.y <= -MOUSE_SENSITIVITY)
		{
			pitchIncrement = vml_angle_axis(rotationSpeed * (float)deltaTime, vml_vec3(1.0f, 0.0f, 0.0f));
		}

		yaw = yawIncrement * yaw;
		pitch = pitchIncrement * pitch;

		camera.localTransform.rotation = yaw * pitch;

		lastMousePosition = mousePosition;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline bool getKey(unsigned int keyCode) const
	{
		return back.keys[keyCode];
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline bool getKeyUp(unsigned int keyCode) const
	{
		return keysUp[keyCode];
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline bool getKeyDown(unsigned int keyCode) const
	{
		return keysDown[keyCode];
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline bool getRightMouseButton() const
	{
		return back.rightMouseButton;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline bool getRightMouseButtonUp() const
	{
		return rightMouseButtonUp;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline bool getRightMouseButtonDown() const
	{
		return rightMouseButtonDown;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline bool getLeftMouseButton() const
	{
		return back.leftMouseButton;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline bool getLeftMouseButtonUp() const
	{
		return leftMouseButtonUp;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline bool getLeftMouseButtonDown() const
	{
		return leftMouseButtonDown;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline vml_vec2 getMousePosition() const
	{
		return back.mousePosition;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline void moveCameraLeft(float deltaTime)
	{
		camera.localTransform.position -= camera.localTransform.right() * moveSpeed * deltaTime;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline void moveCameraRight(float deltaTime)
	{
		camera.localTransform.position += camera.localTransform.right() * moveSpeed * deltaTime;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline void moveCameraForward(float deltaTime)
	{
		camera.localTransform.position += camera.localTransform.forward() * moveSpeed * deltaTime;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline void moveCameraBackward(float deltaTime)
	{
		camera.localTransform.position -= camera.localTransform.forward() * moveSpeed * deltaTime;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline void moveCameraUp(float deltaTime)
	{
		camera.localTransform.position += vml_vec3(0, 1, 0) * moveSpeed * deltaTime;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline void moveCameraDown(float deltaTime)
	{
		camera.localTransform.position -= vml_vec3(0, 1, 0) * moveSpeed * deltaTime;
	}

private:
	struct InputBuffer
	{
		bool keys[NUMBER_OF_VIRTUAL_KEYS];
		bool rightMouseButton;
		bool leftMouseButton;
		vml_vec2 mousePosition;

		InputBuffer() : rightMouseButton(false), leftMouseButton(false)
		{
			memset(keys, 0, NUMBER_OF_VIRTUAL_KEYS * sizeof(bool));
		}

		InputBuffer& operator = (InputBuffer& other)
		{
			memcpy(keys, other.keys, NUMBER_OF_VIRTUAL_KEYS * sizeof(bool));
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
	bool rightMouseButtonUp;
	bool rightMouseButtonDown;
	bool leftMouseButtonUp;
	bool leftMouseButtonDown;
	bool keysUp[NUMBER_OF_VIRTUAL_KEYS];
	bool keysDown[NUMBER_OF_VIRTUAL_KEYS];
	vml_vec2 lastMousePosition;

	////////////////////////////////////////////////////////////////////////////////////////////////////
	void clearTemporaryRegisters()
	{
		leftMouseButtonUp = leftMouseButtonDown = rightMouseButtonUp = rightMouseButtonDown = false;
		memset(keysUp, 0, NUMBER_OF_VIRTUAL_KEYS * sizeof(bool));
		memset(keysDown, 0, NUMBER_OF_VIRTUAL_KEYS * sizeof(bool));
	}

};


#endif