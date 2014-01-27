#ifndef CIRCLE2D_H
#define CIRCLE2D_H

#include "Defines.h"
#include <vector_math.h>

HOST_AND_DEVICE_CODE struct Circle2D
{
	//vml_vec2 center;
	vec2FieldDeclaration(Center, HOST_AND_DEVICE_CODE)
	float radius;

	HOST_AND_DEVICE_CODE Circle2D() : radius(0) {}
	HOST_AND_DEVICE_CODE Circle2D(const vml_vec2& center, float radius) : radius(radius) { setCenter(center); }
	HOST_AND_DEVICE_CODE ~Circle2D() {}

	HOST_AND_DEVICE_CODE Circle2D& operator = (const Circle2D& other)
	{
		setCenter(other.getCenter());
		radius = other.radius;
		return *this;
	}

	HOST_AND_DEVICE_CODE bool contains(const vml_vec2& point) const
	{
		return vml_distance(getCenter(), point) <= radius;
	}


};

#endif