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

	/* ========================================================================================================
	 * Based on: http://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect
	 * ======================================================================================================== */

	HOST_AND_DEVICE_CODE bool intersects(const Line2D& line, vml_vec2& intersection) const
	{
		vml_vec2 p1 = getStart(); 
		vml_vec2 p2 = getEnd(); 
		vml_vec2 q1 = line.getStart();
		vml_vec2 q2 = line.getEnd();

		vml_vec2 r = p2 - p1;
		vml_vec2 s = q2 - q1;

		float uNumerator = vml_dot_perp(q1 - p1, r);
		float denominator = vml_dot_perp(r, s);

		if (uNumerator == 0 && denominator == 0) 
		{
			// collinear, so do they overlap?
			return ((q1.x - p1.x < 0) != (q1.x - p2.x < 0) != (q2.x - p1.x < 0) != (q2.x - p2.x < 0)) || 
				   ((q1.y - p1.y < 0) != (q1.y - p2.y < 0) != (q2.y - p1.y < 0) != (q2.y - p2.y < 0));
		}

		if (denominator == 0) 
		{
			// lines are parallel
			return false;
		}

		float u = uNumerator / denominator;
		float t = vml_dot_perp((q1 - p1), s) / denominator;

		if (t < 0 || t > 1 || u < 0 || u > 1)
		{
			return false;
		}
		else
		{
			intersection = p1 + t * r;
			return true;
		}
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
