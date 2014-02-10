#ifndef BOX2D_H
#define BOX2D_H

#pragma once

#include <CpuGpuCompatibility.h>
#include <Line2D.h>
#include <MathExtras.h>
#include <VectorMath.h>

struct Box2D
{
	vec2FieldDeclaration(Min, HOST_AND_DEVICE_CODE)
	vec2FieldDeclaration(Max, HOST_AND_DEVICE_CODE)

	HOST_AND_DEVICE_CODE Box2D() {}
	HOST_AND_DEVICE_CODE Box2D(const vml_vec2& min, const vml_vec2& max) { x_Min = min.x; y_Min = min.y; x_Max = max.x; y_Max = max.y; }
	HOST_AND_DEVICE_CODE Box2D(float x, float y, float width, float height)
	{
		x_Min = x; y_Min = y;
		x_Max = x + width; y_Max = y + height;
	}
	HOST_AND_DEVICE_CODE Box2D(const Line2D& line)
	{
		setMin(MathExtras::min(line.getStart(), line.getEnd()));
		setMax(MathExtras::max(line.getStart(), line.getEnd()));
	}
	HOST_AND_DEVICE_CODE ~Box2D() {}

	inline HOST_AND_DEVICE_CODE vml_vec2 getExtents() const
	{
		return getMax() - getMin();
	}

	inline HOST_AND_DEVICE_CODE vml_vec2 getCenter() const
	{
		return getMin() + ((getMax() - getMin()) / 2.0f);
	}

	inline HOST_AND_DEVICE_CODE bool contains(const vml_vec2& point) const
	{
		if (point.x < x_Min)
		{
			return false;
		}

		else if (point.x > x_Max)
		{
			return false;
		}

		else if (point.y < y_Min)
		{
			return false;
		}

		else if (point.y > y_Max)
		{
			return false;
		}

		else
		{
			return true;
		}
	}

	HOST_AND_DEVICE_CODE Box2D& operator = (const Box2D& other)
	{
		x_Min = other.x_Min; y_Min = other.y_Min;
		x_Max = other.x_Max; y_Max = other.y_Max;
		return *this;
	}

	inline HOST_AND_DEVICE_CODE float getArea() const
	{
		vml_vec2 size = getExtents();
		return size.x * size.y;
	}

	/*inline HOST_AND_DEVICE_CODE bool intersects(const Circle2D& circle) const
	{
		vml_vec2 a(x_Min, y_Max);
		vml_vec2 b(x_Max, y_Max);
		vml_vec2 c(x_Max, y_Min);
		vml_vec2 d(x_Min, y_Min);
		Line2D AB(a, b);
		Line2D BC(b, c);
		Line2D CD(c, d);
		Line2D DA(d, a);
		vml_vec2 intersection1;
		vml_vec2 intersection2;
		return contains(circle.getCenter()) || AB.intersects(circle, intersection1, intersection2) || BC.intersects(circle, intersection1, intersection2) || CD.intersects(circle, intersection1, intersection2) || DA.intersects(circle, intersection1, intersection2);
	}

	inline HOST_AND_DEVICE_CODE bool intersects(const Box2D& aabb) const
	{
		if (aabb.x_Max < x_Min || aabb.x_Min > x_Max || aabb.y_Max < y_Min || aabb.y_Min > y_Max)
		{
			return false;
		}

		return true;
	}*/

	inline HOST_AND_DEVICE_CODE bool isIntersected(const Line2D& line) const
	{
		vml_vec2 a(x_Min, y_Max);
		vml_vec2 b(x_Max, y_Max);
		vml_vec2 c(x_Max, y_Min);
		vml_vec2 d(x_Min, y_Min);
		
		if (contains(line.getStart()))
		{
			return true;
		}
		
		if (contains(line.getEnd()))
		{
			return true;
		}
		
		Line2D AB(a, b);
		Line2D BC(b, c);
		Line2D CD(c, d);
		Line2D DA(d, a);
		vml_vec2 intersection;

		if (AB.intersects(line, intersection))
		{
			return true;
		}
		
		if (BC.intersects(line, intersection))
		{
			return true;
		}
		
		if (CD.intersects(line, intersection))
		{
			return true;
		}
		
		if (DA.intersects(line, intersection))
		{
			return true;
		}
		
		return false;
	}

};

#endif