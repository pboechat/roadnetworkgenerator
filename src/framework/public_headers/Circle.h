#ifndef CIRCLE_H
#define CIRCLE_H

#include <glm/glm.hpp>
#include <glm/gtx/projection.hpp>

struct Circle
{
	glm::vec3 center;
	float radius;

	Circle(const glm::vec3& center, float radius) : center(center), radius(radius) {}
	~Circle() {}

	Circle& operator = (const Circle& other)
	{
		center = other.center;
		radius = other.radius;
		return *this;
	}

	bool contains(const glm::vec3& point) const
	{
		return glm::distance(center, point) <= radius;
	}


};

#endif