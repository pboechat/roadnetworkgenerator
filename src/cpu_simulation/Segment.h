#ifndef SEGMENT_H
#define SEGMENT_H

#include <glm/glm.hpp>

struct Segment
{
	glm::vec3 start;
	glm::vec3 end;

	Segment(const glm::vec3& start, const glm::vec3& end) : start(start), end(end) {}
	~Segment() {}

	Segment& operator = (const Segment& other)
	{
		start = other.start;
		end = other.end;
		return *this;
	}
};

#endif