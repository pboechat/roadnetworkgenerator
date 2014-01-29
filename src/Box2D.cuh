#ifndef BOX2D_CUH
#define BOX2D_CUH

#include "Defines.h"
#include <Line2D.cuh>
#include <MathExtras.cuh>

#include <vector_math.h>

struct Box2D
{
	vec2FieldDeclaration(Min, HOST_AND_DEVICE_CODE)
	vec2FieldDeclaration(Max, HOST_AND_DEVICE_CODE)

	HOST_AND_DEVICE_CODE Box2D() {}
	HOST_AND_DEVICE_CODE Box2D(const vml_vec2& min, const vml_vec2& max) { setMin(min); setMax(max); }
	HOST_AND_DEVICE_CODE Box2D(float x, float y, float width, float height)
	{
		setMin(vml_vec2(x, y));
		setMax(vml_vec2(x + width, y + height));
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
		if (point.x < getMin().x)
		{
			return false;
		}

		else if (point.x > getMax().x)
		{
			return false;
		}

		else if (point.y < getMin().y)
		{
			return false;
		}

		else if (point.y > getMax().y)
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
		setMin(other.getMin());
		setMax(other.getMax());
		return *this;
	}

	inline HOST_AND_DEVICE_CODE float getArea() const
	{
		vml_vec2 size = getExtents();
		return size.x * size.y;
	}

	HOST_AND_DEVICE_CODE bool intersects(const Circle2D& circle) const
	{
		vml_vec2 a(getMin().x, getMax().y);
		vml_vec2 b(getMax().x, getMax().y);
		vml_vec2 c(getMax().x, getMin().y);
		vml_vec2 d(getMin().x, getMin().y);
		return contains(circle.getCenter()) || Line2D(a, b).intersects(circle) || Line2D(b, c).intersects(circle) || Line2D(c, d).intersects(circle) || Line2D(d, a).intersects(circle);
	}

	HOST_AND_DEVICE_CODE bool intersects(const Box2D& aabb) const
	{
		if (aabb.getMax().x < getMin().x || aabb.getMin().x > getMax().x || aabb.getMax().y < getMin().y || aabb.getMin().y > getMax().y)
		{
			return false;
		}

		return true;
	}

	HOST_AND_DEVICE_CODE bool isIntersected(const Line2D& line) const
	{
		vml_vec2 a(getMin().x, getMax().y);
		vml_vec2 b(getMax().x, getMax().y);
		vml_vec2 c(getMax().x, getMin().y);
		vml_vec2 d(getMin().x, getMin().y);
		return contains(line.getStart()) || contains(line.getEnd()) || Line2D(a, b).intersects(line) || Line2D(b, c).intersects(line) || Line2D(c, d).intersects(line) || Line2D(d, a).intersects(line);
	}

};

#endif
