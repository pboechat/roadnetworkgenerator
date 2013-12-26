#ifndef CIRCLE2D_H
#define CIRCLE2D_H

#include <vector_math.h>

struct Circle2D
{
	vml_vec2 center;
	float radius;

	Circle2D(const vml_vec2& center, float radius) : center(center), radius(radius) {}
	~Circle2D() {}

	Circle2D& operator = (const Circle2D& other)
	{
		center = other.center;
		radius = other.radius;
		return *this;
	}

	bool contains(const vml_vec2& point) const
	{
		return vml_distance(center, point) <= radius;
	}


};

#endif