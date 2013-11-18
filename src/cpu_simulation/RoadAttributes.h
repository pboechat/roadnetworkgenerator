#ifndef ROADATTRIBUTES_H
#define ROADATTRIBUTES_H

struct RoadAttributes
{
	glm::vec3 start;
	int length;
	float angle;
	bool highway;

	RoadAttributes() {}
	RoadAttributes(const glm::vec3& start, int length, float angle, bool highway) : start(start), length(length), angle(angle), highway(highway) {}
	~RoadAttributes() {}

	RoadAttributes& operator =(const RoadAttributes& other)
	{
		start = other.start;
		length = other.length;
		angle = other.angle;
		highway = other.highway;
		return *this;
	}

};

#endif