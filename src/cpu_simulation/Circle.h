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

	bool intersect(const Line& line) const
	{
		glm::vec3 ac = center - line.start;
		glm::vec3 ab = line.end - line.start;
		glm::vec3 d = line.start + glm::proj(ac, ab);
		return (glm::distance(d, center) <= radius);
	}


};

#endif