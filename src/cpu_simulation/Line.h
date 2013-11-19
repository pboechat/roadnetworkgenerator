#ifndef LINE_H
#define LINE_H

#include <Circle.h>

#define _USE_MATH_DEFINES
#include <math.h>

#include <glm/glm.hpp>
#include <glm/gtx/projection.hpp>

struct Line
{
	glm::vec3 start;
	glm::vec3 end;
	int thickness;
	glm::vec4 color1;
	glm::vec4 color2;

	Line(const glm::vec3& start, const glm::vec3& end) : start(start), end(end) {}
	Line(const glm::vec3& start, const glm::vec3& end, int thickness, const glm::vec4& color1, const glm::vec4& color2) : start(start), end(end), thickness(thickness), color1(color1), color2(color2) {}
	~Line() {}

	Line& operator = (const Line& other)
	{
		start = other.start;
		end = other.end;
		thickness = other.thickness;
		color1 = other.color1;
		color2 = other.color2;
		return *this;
	}

	inline float orientedAngle(const glm::vec3& a, const glm::vec3& b) const
	{
		float angle = glm::acos(glm::dot(a, b) / (glm::length(a) * glm::length(b)));

		if (glm::cross(a, b).z >= 0)
		{
			return angle;
		}
		else
		{
			return (float)(2.0f * M_PI) - angle;
		}
	}

	inline glm::vec3 snap(const glm::vec3& point) const
	{
		glm::vec3 a = point - start;
		glm::vec3 n = end - start;

		if (glm::length(a) > glm::length(n)) 
		{
			return end;
		}
		else if (glm::degrees(orientedAngle(a, n)) >= 90)
		{
			return start;
		}

		return start + glm::proj(a, n);
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

		float s = ((x1 * y2 - y1 * x2) * (x3 - x4)) - ((x1 - x2) * (x3 * y4 - y3 * x4));
		float t = ((x1 * y2 - y1 * x2) * (y3 - y4)) - ((y1 - y2) * (x3 * y4 - y3 * x4));

		intersection.x = s / determinant;
		intersection.y = t / determinant;

		return true;
	}

	bool intersects(const Circle& circle) const
	{
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

};

#endif