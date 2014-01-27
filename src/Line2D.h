#ifndef LINE2D_H
#define LINE2D_H

#include "Defines.h"
#include <Circle2D.h>
#include <MathExtras.h>

#include <vector_math.h>

HOST_AND_DEVICE_CODE struct Line2D
{
	//vml_vec2 getStart();
	//vml_vec2 getEnd();
	vec2FieldDeclaration(Start, HOST_AND_DEVICE_CODE)
	vec2FieldDeclaration(End, HOST_AND_DEVICE_CODE)

	HOST_AND_DEVICE_CODE Line2D() {}
	HOST_AND_DEVICE_CODE Line2D(const vml_vec2& start, const vml_vec2& end) { setStart(start); setEnd(end); }
	HOST_AND_DEVICE_CODE ~Line2D() {}

	HOST_AND_DEVICE_CODE Line2D& operator = (const Line2D& other)
	{
		setStart(other.getStart());
		setEnd(other.getEnd());
		return *this;
	}

	HOST_AND_DEVICE_CODE bool intersects(const Line2D& line) const
	{
		vml_vec2 intersection;
		return intersects(line, intersection);
	}

	HOST_AND_DEVICE_CODE inline bool onSegment(const vml_vec2& p, const vml_vec2& q, const vml_vec2& r) const
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
	inline HOST_AND_DEVICE_CODE int orientation(const vml_vec2& p, const vml_vec2& q, const vml_vec2& r) const
	{
		float value = (r.x - q.x) * (q.y - p.y) - (r.y - q.y) * (q.x - p.x);

		if (MathExtras::isZero(value))
		{
			return 0; // collinear
		}

		return (value > 0) ? 1 : 2; // clock or counterclockwise
	}

	HOST_AND_DEVICE_CODE bool intersects(const Line2D& line, vml_vec2& intersection) const
	{
		// find the four orientations needed for general and special cases
		int o1 = orientation(getStart(), getEnd(), line.getStart());
		int o2 = orientation(getStart(), getEnd(), line.getEnd());
		int o3 = orientation(line.getStart(), line.getEnd(), getStart());
		int o4 = orientation(line.getStart(), line.getEnd(), getEnd());

		// general case:
		if (o1 != o2 && o3 != o4)
		{
			float determinant = (getStart().x - getEnd().x) * (line.getStart().y - line.getEnd().y) - (getStart().y - getEnd().y) * (line.getStart().x - line.getEnd().x);

			// FIXME: checking invariants
			 if (determinant == 0)
			{
				//throw std::exception("determinant == 0");
				THROW_EXCEPTION("determinant == 0");
			}

			float pre = (getStart().x * getEnd().y - getStart().y * getEnd().x), post = (line.getStart().x * line.getEnd().y - line.getStart().y * line.getEnd().x);
			float x = (pre * (line.getStart().x - line.getEnd().x) - (getStart().x - getEnd().x) * post) / determinant;
			float y = (pre * (line.getStart().y - line.getEnd().y) - (getStart().y - getEnd().y) * post) / determinant;

			intersection.x = x;
			intersection.y = y;

			return true;
		}

		// special cases:
		// 'getStart()', 'getEnd()' and 'line.getStart()' are collinear and 'line.getStart()' lies on this segment
		if (o1 == 0 && onSegment(getStart(), line.getStart(), getEnd()))
		{
			intersection = line.getStart();
			return true;
		}

		// 'getStart()', 'getEnd()' and 'line.getStart()' are collinear and 'line.getEnd()' lies on this segment
		if (o2 == 0 && onSegment(getStart(), line.getEnd(), getEnd()))
		{
			intersection = line.getEnd();
			return true;
		}

		// 'line.getStart()', 'line.getEnd()' and 'getStart()' are collinear and 'getStart()' lies on line segment
		if (o3 == 0 && onSegment(line.getStart(), getStart(), line.getEnd()))
		{
			intersection = getStart();
			return true;
		}

		// 'line.getStart()', 'line.getEnd()' and 'getEnd()' are collinear and 'getEnd()' lies on line segment
		if (o4 == 0 && onSegment(line.getStart(), getEnd(), line.getEnd()))
		{
			intersection = getEnd();
			return true;
		}

		return false;
	}

	HOST_AND_DEVICE_CODE unsigned int intersects(const Circle2D& circle, vml_vec2& intersection1, vml_vec2& intersection2) const
	{
		// FIXME: circle == point case
		//if (circle.radius == 0)
		if (MathExtras::isZero(circle.radius))
		{
			return 0;
		}

		vml_vec2 direction = vml_normalize(getEnd() - getStart());
		vml_vec2 centerToStart = getStart() - circle.getCenter();
		float a = vml_dot(direction, direction);
		float b = 2.0f * vml_dot(centerToStart, direction);
		float c = vml_dot(centerToStart, centerToStart) - circle.radius * circle.radius;
		float discriminant = b * b - 4 * a * c;

		if (discriminant < 0)
		{
			return 0;
		}

		else
		{
			unsigned int mask = 0;
			discriminant = sqrt(discriminant);
			float t1 = (-b - discriminant) / (2.0f * a);
			float t2 = (-b + discriminant) / (2.0f * a);

			if (t1 >= 0 && t1 <= 1)
			{
				intersection1 = getStart() + direction * t1;
				mask += 1;
			}

			if (t2 >= 0 && t2 <= 1)
			{
				intersection2 = getStart() + direction * t2;
				mask += 2;
			}

			return mask;
		}
	}

	bool intersects(const Circle2D& circle) const
	{
		vml_vec2 i1, i2;
		return intersects(circle, i1, i2) > 0;
	}

};

#endif
