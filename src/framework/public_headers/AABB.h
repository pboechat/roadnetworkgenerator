#ifndef AABB_H
#define AABB_H

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

	inline glm::vec3 extents() const
	{
		return max - min;
	}

	inline glm::vec3 center() const
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

	inline float area() const
	{
		 glm::vec3 size = extents();
		 return size.x * size.y;
	}


};

#endif
