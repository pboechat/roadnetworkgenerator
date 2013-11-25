#ifndef CAMERA_H
#define CAMERA_H

#include <Transform.h>
#include <AABB.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

struct Camera
{
public:
	Transform localTransform;

	////////////////////////////////////////////////////////////////////////////////////////////////////
	Camera::Camera(unsigned int screenWidth, unsigned int screenHeight, float fovY, float _near, float _far) :
		screenWidth(screenWidth),
		screenHeight(screenHeight),
		aspectRatio(screenWidth / (float)screenHeight),
		fovY(fovY),
		_near(_near),
		_far(_far)
	{
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline unsigned int getScreenWidth() const
	{
		return screenWidth;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline unsigned int getScreenHeight() const
	{
		return screenHeight;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline float getFovY() const
	{
		return fovY;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline float getFar() const
	{
		return _far;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline glm::vec3 getPosition() const
	{
		return worldTransform.position;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline glm::mat4 getProjectionMatrix() const
	{
		return glm::perspective(fovY, aspectRatio, _near, _far);
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline const glm::mat4 getViewMatrix() const
	{
		return glm::lookAt(worldTransform.position, worldTransform.position + worldTransform.forward(), worldTransform.up());
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	void centerOnTarget(const AABB& target)
	{
		glm::vec3 size = target.getExtents();
		float diagonal = glm::sqrt(glm::pow(size.x, 2.0f) + glm::pow(size.y, 2.0f));
		float distance = (diagonal / 2.0f) / glm::tan(glm::radians(fovY / 2.0f));
		localTransform.position = glm::vec3(target.min.x + size.x / 2.0f, target.min.y + size.y / 2.0f, distance);
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	void update(double elapsedTime)
	{
		worldTransform = localTransform;
	}

private:
	unsigned int screenWidth;
	unsigned int screenHeight;
	float aspectRatio;
	float fovY;
	float _near;
	float _far;
	Transform worldTransform;

};

#endif