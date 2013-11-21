#ifndef LINE_H
#define LINE_H

#define EPSILON 0.0001f

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

		float x = ((x1 * y2 - y1 * x2) * (x3 - x4)) - ((x1 - x2) * (x3 * y4 - y3 * x4)) / determinant;
		float y = ((x1 * y2 - y1 * x2) * (y3 - y4)) - ((y1 - y2) * (x3 * y4 - y3 * x4)) / determinant;

		if (x < glm::min(x1, x2) || x > glm::max(x1, x2) || x < glm::min(x3, x4) || x > glm::max(x3, x4)) 
		{
			return false;
		}

		if (y < glm::min(y1, y2) || y > glm::max(y1, y2) || y < glm::min(y3, y4) || y > glm::max(y3, y4)) 
		{
			return false;
		}

		intersection.x = x;
		intersection.y = y;

		return true;
	}

	bool intersects(const Circle& circle) const
	{
		// FIXME:
		if (circle.radius == 0)
		{
			return false;
		}

		glm::vec3 d = end - start;
		glm::vec3 f = start - circle.center;

		float a = glm::dot(d, d);
		float b = 2.0f * glm::dot(f, d);
		float c = glm::dot(f, f) - circle.radius * circle.radius;

		float discriminant = b * b - 4 * a * c;
		if (discriminant < 0)
		{
			return false;
		}
		else
		{
			discriminant = glm::sqrt(discriminant);

			float t1 = (-b - discriminant) / (2.0f * a);
			float t2 = (-b + discriminant) / (2.0f * a);

			if (t1 >= 0 && t1 <= 1)
			{
				return true;
			}

			if (t2 >= 0 && t2 <= 1)
			{
				return true;
			}

			return false;
		}
	}

	bool contains(const glm::vec3& point) const
	{
		// get the normalized line segment vector
		glm::vec3 v = glm::normalize(end - start);

		// determine the point on the line segment nearest to the point p
		float distanceAlongLine = glm::dot(point, v) - glm::dot(start, v);
		glm::vec3 nearestPoint;
		if (distanceAlongLine < 0)
		{
			// closest point is A
			nearestPoint = start;
		}
		else if (distanceAlongLine > glm::distance(start, end))
		{
			// closest point is B
			nearestPoint = end;
		}
		else
		{
			// closest point is between A and B... A + d  * ( ||B-A|| )
			nearestPoint = start + distanceAlongLine * v;  
		}

		// calculate the distance between the two points
		return (glm::distance(nearestPoint, point) <= EPSILON);
	}


};

#endif