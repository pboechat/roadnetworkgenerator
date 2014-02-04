#ifndef LINE2D_H
#define LINE2D_H

#pragma once

#include <CpuGpuCompatibility.h>
#include <Circle2D.h>
#include <MathExtras.h>
#include <VectorMath.h>

struct Line2D
{
	vec2FieldDeclaration(Start, HOST_AND_DEVICE_CODE)
	vec2FieldDeclaration(End, HOST_AND_DEVICE_CODE)

	HOST_AND_DEVICE_CODE Line2D() {}
	HOST_AND_DEVICE_CODE Line2D(const vml_vec2& start, const vml_vec2& end) { x_Start = start.x; y_Start = start.y; x_End = end.x; y_End = end.y; }
	HOST_AND_DEVICE_CODE ~Line2D() {}

	HOST_AND_DEVICE_CODE Line2D& operator = (const Line2D& other)
	{
		x_Start = other.x_Start; y_Start = other.y_Start;
		x_End = other.x_End; y_End = other.y_End;
		return *this;
	}

	inline HOST_AND_DEVICE_CODE bool onSegment(const vml_vec2& p, const vml_vec2& q, const vml_vec2& r) const
	{
		if (q.x <= MathExtras::max(p.x, r.x) && q.x >= MathExtras::min(p.x, r.x) &&
			q.y <= MathExtras::max(p.y, r.y) && q.y >= MathExtras::min(p.y, r.y))
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
		vml_vec2 s1 = getStart(); vml_vec2 s2 = line.getStart();
		vml_vec2 e1 = getEnd(); vml_vec2 e2 = line.getEnd();

		// find the four orientations needed for general and special cases
		int o1 = orientation(s1, e1, s2);
		int o2 = orientation(s1, e1, e2);
		int o3 = orientation(s2, e2, s1);
		int o4 = orientation(s2, e2, e1);

		// general case:
		if (o1 != o2 && o3 != o4)
		{
			float determinant = (s1.x - e1.x) * (s2.y - e2.y) - (s1.y - e1.y) * (s2.x - e2.x);

			// FIXME: checking invariants
			if (determinant == 0)
			{
				THROW_EXCEPTION("determinant == 0");
			}

			float pre = (s1.x * e1.y - s1.y * e1.x), post = (s2.x * e2.y - s2.y * e2.x);
			float x = (pre * (s2.x - e2.x) - (s1.x - e1.x) * post) / determinant;
			float y = (pre * (s2.y - e2.y) - (s1.y - e1.y) * post) / determinant;

			intersection.x = x;
			intersection.y = y;

			return true;
		}

		// special cases:
		// 's1', 'e1' and 's2' are collinear and 's2' lies on this segment
		if (o1 == 0 && onSegment(s1, s2, e1))
		{
			intersection = s2;
			return true;
		}

		// 's1', 'e1' and 's2' are collinear and 'e2' lies on this segment
		if (o2 == 0 && onSegment(s1, e2, e1))
		{
			intersection = e2;
			return true;
		}

		// 's2', 'e2' and 's1' are collinear and 's1' lies on line segment
		if (o3 == 0 && onSegment(s2, s1, e2))
		{
			intersection = s1;
			return true;
		}

		// 's2', 'e2' and 'e1' are collinear and 'e1' lies on line segment
		if (o4 == 0 && onSegment(s2, e1, e2))
		{
			intersection = e1;
			return true;
		}

		return false;
	}

	HOST_AND_DEVICE_CODE unsigned int intersects(const Circle2D& circle, vml_vec2& intersection1, vml_vec2& intersection2) const
	{
		// FIXME: circle == point case
		if (MathExtras::isZero(circle.radius))
		{
			return 0;
		}

		vml_vec2 start = getStart();
		vml_vec2 end = getEnd();
		vml_vec2 direction = vml_normalize(end - start);
		vml_vec2 centerToStart = start - circle.getCenter();
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

};

#endif
