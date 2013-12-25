#ifndef LINE_H
#define LINE_H

#include <Circle.h>

#include <glm/glm.hpp>
#include <glm/gtx/projection.hpp>

#include <exception>

#define EPSILON 0.001f

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

	bool intersects(const Line& line) const
	{
		glm::vec3 intersection;
		return intersects(line, intersection);
	}

	inline bool onSegment(const glm::vec3& p, const glm::vec3& q, const glm::vec3& r) const
	{
		if (q.x <= glm::max(p.x, r.x) && q.x >= glm::min(p.x, r.x) &&
				q.y <= glm::max(p.y, r.y) && q.y >= glm::min(p.y, r.y))
		{
			return true;
		}

		return false;
	}

	// 0 -> collinear
	// 1 -> clockwise
	// 2 -> counterclockwise
	inline int orientation(const glm::vec3& p, const glm::vec3& q, const glm::vec3& r) const
	{
		float val = (r.x - q.x) * (q.y - p.y) - (r.y - q.y) * (q.x - p.x);

		if (val >= -EPSILON && val <= EPSILON)
		{
			return 0;    // collinear
		}

		return (val > 0) ? 1 : 2; // clock or counterclockwise
	}

	bool intersects(const Line& line, glm::vec3& intersection) const
	{
		// find the four orientations needed for general and special cases
		int o1 = orientation(start, end, line.start);
		int o2 = orientation(start, end, line.end);
		int o3 = orientation(line.start, line.end, start);
		int o4 = orientation(line.start, line.end, end);

		// general case:
		if (o1 != o2 && o3 != o4)
		{
			float determinant = (start.x - end.x) * (line.start.y - line.end.y) - (start.y - end.y) * (line.start.x - line.end.x);

			// FIXME: checking invariants
			if (determinant == 0)
			{
				throw std::exception("determinant == 0");
			}

			float pre = (start.x * end.y - start.y * end.x), post = (line.start.x * line.end.y - line.start.y * line.end.x);
			float x = (pre * (line.start.x - line.end.x) - (start.x - end.x) * post) / determinant;
			float y = (pre * (line.start.y - line.end.y) - (start.y - end.y) * post) / determinant;
			intersection.x = x;
			intersection.y = y;
			return true;
		}

		// special cases:
		// 'start', 'end' and 'line.start' are collinear and 'line.start' lies on this segment
		if (o1 == 0 && onSegment(start, line.start, end))
		{
			intersection = line.start;
			return true;
		}

		// 'start', 'end' and 'line.start' are collinear and 'line.end' lies on this segment
		if (o2 == 0 && onSegment(start, line.end, end))
		{
			intersection = line.end;
			return true;
		}

		// 'line.start', 'line.end' and 'start' are collinear and 'start' lies on line segment
		if (o3 == 0 && onSegment(line.start, start, line.end))
		{
			intersection = start;
			return true;
		}

		// 'line.start', 'line.end' and 'end' are collinear and 'end' lies on line segment
		if (o4 == 0 && onSegment(line.start, end, line.end))
		{
			intersection = end;
			return true;
		}

		return false;
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