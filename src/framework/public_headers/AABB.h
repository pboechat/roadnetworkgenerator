#ifndef AABB_H
#define AABB_H

#include <Line.h>

#include <glm/glm.hpp>

struct AABB
{
	glm::vec3 max;
	glm::vec3 min;

	AABB() {}

	AABB(float x, float y, float width, float height)
	{
		min = glm::vec3(x, y, 0);
		max = glm::vec3(x + width, y + height, 0);
	}

	~AABB() {}

	inline glm::vec3 getExtents() const
	{
		return max - min;
	}

	inline glm::vec3 getCenter() const
	{
		return min + ((max - min) / 2.0f);
	}

	inline bool contains(const glm::vec3& point) const
	{
		if (point.x < min.x)
		{
			return false;
		}

		else if (point.x > max.x)
		{
			return false;
		}

		else if (point.y < min.y)
		{
			return false;
		}

		else if (point.y > max.y)
		{
			return false;
		}

		else
		{
			return true;
		}
	}

	AABB& operator = (const AABB& other)
	{
		min = other.min;
		max = other.max;
		return *this;
	}

	inline float getArea() const
	{
		 glm::vec3 size = getExtents();
		 return size.x * size.y;
	}

	bool intersects(const Circle& circle) const
	{
		glm::vec3 a(min.x, max.y, 0.0f);
		glm::vec3 b(max.x, max.y, 0.0f);
		glm::vec3 c(max.x, min.y, 0.0f);
		glm::vec3 d(min.x, min.y, 0.0f);

		return contains(circle.center) || Line(a, b).intersects(circle) || Line(b, c).intersects(circle) || Line(c, d).intersects(circle) || Line(d, a).intersects(circle);
	}

};

#endif
