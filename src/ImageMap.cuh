#ifndef IMAGEMAP_CUH
#define IMAGEMAP_CUH

#include "Defines.h"
#include <MathExtras.cuh>

#include <vector_math.h>

struct ImageMap
{
	HOST_AND_DEVICE_CODE ImageMap() : width(0), height(0), data(0), color1(0.0f, 0.0f, 0.0f, 1.0f), color2(1.0f, 1.0f, 1.0f, 1.0f) {}
	HOST_AND_DEVICE_CODE ~ImageMap() {}

	inline HOST_AND_DEVICE_CODE bool hasData() const
	{
		return data != 0;
	}

	inline HOST_AND_DEVICE_CODE unsigned int getWidth() const
	{
		return width;
	}

	inline HOST_AND_DEVICE_CODE unsigned int getHeight() const
	{
		return height;
	}

	inline HOST_AND_DEVICE_CODE const unsigned char* getData() const
	{
		return data;
	}
	
	HOST_AND_DEVICE_CODE bool castRay(const vml_vec2& origin, const vml_vec2& direction, unsigned int length, unsigned char threshold, vml_vec2& hit) const
	{
		for (unsigned int i = 0; i <= length; i++)
		{
			vml_vec2 point = origin + (direction * (float)i);

			if (sample(point) > threshold)
			{
				hit = point;
				return false;
			}
		}

		return true;
	}

	HOST_AND_DEVICE_CODE bool castRay(const vml_vec2& origin, const vml_vec2& direction, unsigned int length, unsigned char threshold) const
	{
		vml_vec2 hit;
		return castRay(origin, direction, length, threshold, hit);
	}

	HOST_AND_DEVICE_CODE void scan(const vml_vec2& origin, const vml_vec2& direction, int minDistance, int maxDistance, unsigned char& greaterSample, int& distance) const
	{
		distance = maxDistance;
		greaterSample = 0;

		for (int i = minDistance; i <= maxDistance; i++)
		{
			vml_vec2 point = origin + (direction * (float)i);
			unsigned char currentSample = sample(point);

			if (currentSample > greaterSample)
			{
				distance = i;
				greaterSample = currentSample;
			}
		}
	}

	HOST_AND_DEVICE_CODE unsigned char sample(const vml_vec2& point) const
	{
		vml_vec2 position;
		position.x = MathExtras::clamp(0.0f, (float)width, point.x);
		position.y = MathExtras::clamp(0.0f, (float)height, point.y);
		int i = ((int)position.y * width) + (int)position.x;
		return data[i];
	}

private:
	unsigned int width;
	unsigned int height;
	const unsigned char* data;
	vml_vec4 color1;
	vml_vec4 color2;

};

#endif