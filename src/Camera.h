#ifndef CAMERA_H
#define CAMERA_H

#include <Transform.h>
#include <Box2D.h>

#include <vector_math.h>

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
	inline vml_vec3 getPosition() const
	{
		return worldTransform.position;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline vml_mat4 getProjectionMatrix() const
	{
		return vml_perspective(fovY, aspectRatio, _near, _far);
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	inline vml_mat4 getViewMatrix() const
	{
		return vml_look_at(worldTransform.position, worldTransform.position + worldTransform.forward(), worldTransform.up());
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	void centerOnTarget(const Box2D& target)
	{
		vml_vec2 size = target.getExtents();
		float diagonal = sqrt(pow(size.x, 2.0f) + pow(size.y, 2.0f));
		float distance = (diagonal / 2.0f) / tan(vml_radians(fovY / 2.0f));
		localTransform.position = vml_vec3(target.getMin().x + size.x / 2.0f, target.getMin().y + size.y / 2.0f, distance);
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