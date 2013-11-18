#ifndef AABB_H
#define AABB_H

#include <glm/glm.hpp>

struct AABB
{
	glm::vec3 max;
	glm::vec3 min;
	
	inline glm::vec3 extents() const
	{
		return max - min;
	}

	inline glm::vec3 center() const
	{
		return min + ((max - min) / 2.0f);
	}
};

#endif
