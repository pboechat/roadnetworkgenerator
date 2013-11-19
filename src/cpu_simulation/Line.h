#ifndef LINE_H
#define LINE_H

#include <glm/glm.hpp>
#include <glm/gtx/projection.hpp>

struct Line
{
	glm::vec3 start;
	glm::vec3 end;
	int thickness;

	Line(const glm::vec3& start, const glm::vec3& end, int thickness) : start(start), end(end), thickness(thickness) {}
	~Line() {}

	Line& operator = (const Line& other)
	{
		start = other.start;
		end = other.end;
		thickness = other.thickness;
		return *this;
	}

	inline glm::vec3 snap(const glm::vec3& p) const
	{
		return start + glm::proj(p - start, end - start);
	}

};

#endif