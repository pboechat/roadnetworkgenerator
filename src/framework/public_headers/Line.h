#ifndef LINE_H
#define LINE_H

#include <Circle.h>

#include <glm/glm.hpp>
#include <glm/gtx/projection.hpp>

struct Line
{
	glm::vec3 start;
	glm::vec3 end;

	Line(const glm::vec3& start, const glm::vec3& end) : start(start), end(end) {}
	~Line() {}

	Line& operator = (const Line& other)
	{
		start = other.start;
		end = other.end;
		return *this;
	}

	bool intersects(const Line& line, glm::vec3& intersection) const
	{
		float x1 = start.x;
		float y1 = start.y;
		float x2 = end.x;
		float y2 = end.y;
		float x3 = line.start.x;
		float y3 = line.start.y;
		float x4 = line.end.x;
		float y4 = line.end.y;

		float determinant = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);

		if (determinant == 0)
		{
			return false;
		}

		float pre = (x1 * y2 - y1 * x2), post = (x3 * y4 - y3 * x4);
		float x = (pre * (x3 - x4) - (x1 - x2) * post) / determinant;
		float y = (pre * (y3 - y4) - (y1 - y2) * post) / determinant;

		if (x < glm::min(x1, x2) || x > glm::max(x1, x2) ||
			x < glm::min(x3, x4) || x > glm::max(x3, x4))
		{
			return false;
		}

		if (y < glm::min(y1, y2) || y > glm::max(y1, y2) ||
			y < glm::min(y3, y4) || y > glm::max(y3, y4))
		{
			return false;
		}

		intersection.x = x;
		intersection.y = y;

		return true;
	}

	unsigned int intersects(const Circle& circle, glm::vec3& intersection1, glm::vec3& intersection2) const
	{
		// FIXME: circle == point case
		if (circle.radius == 0)
		{
			return 0;
		}

		glm::vec3 direction = glm::normalize(end - start);
		glm::vec3 centerToStart = start - circle.center;

		float a = glm::dot(direction, direction);
		float b = 2.0f * glm::dot(centerToStart, direction);
		float c = glm::dot(centerToStart, centerToStart) - circle.radius * circle.radius;

		float discriminant = b * b - 4 * a * c;

		if (discriminant < 0)
		{
			return 0;
		}

		else
		{
			unsigned int mask = 0;

			discriminant = glm::sqrt(discriminant);

			float t1 = (-b - discriminant) / (2.0f * a);
			float t2 = (-b + discriminant) / (2.0f * a);

			if (t1 >= 0 && t1 <= 1)
			{
				intersection1 = start + direction * t1;
				mask += 1;
			}

			if (t2 >= 0 && t2 <= 1)
			{
				intersection2 = start + direction * t2;
				mask += 2;
			}

			return mask;
		}
	}

	bool intersects(const Circle& circle) const
	{
		glm::vec3 i1, i2;
		return intersects(circle, i1, i2) > 0;
	}

};

#endif