#ifndef BOX2D_H
#define BOX2D_H

#include <Line2D.h>
#include <MathExtras.h>

#include <vector_math.h>

struct Box2D
{
	vml_vec2 max;
	vml_vec2 min;

	Box2D() {}
	Box2D(const vml_vec2& min, const vml_vec2& max) : min(min), max(max) {}
	Box2D(float x, float y, float width, float height)
	{
		min = vml_vec2(x, y);
		max = vml_vec2(x + width, y + height);
	}
	Box2D(const Line2D& line)
	{
		min = MathExtras::min(line.start, line.end);
		max = MathExtras::max(line.start, line.end);
	}
	~Box2D() {}

	inline vml_vec2 getExtents() const
	{
		return max - min;
	}

	inline vml_vec2 getCenter() const
	{
		return min + ((max - min) / 2.0f);
	}

	inline bool contains(const vml_vec2& point) const
	{
		if (point.x < min.x)
		{
			return false;
		}

		else if (point.x > max.x)
		{
			return false;
		}

		else if (point.y < min.y)
		{
			return false;
		}

		else if (point.y > max.y)
		{
			return false;
		}

		else
		{
			return true;
		}
	}

	Box2D& operator = (const Box2D& other)
	{
		min = other.min;
		max = other.max;
		return *this;
	}

	inline float getArea() const
	{
		vml_vec2 size = getExtents();
		return size.x * size.y;
	}

	bool intersects(const Circle2D& circle) const
	{
		vml_vec2 a(min.x, max.y);
		vml_vec2 b(max.x, max.y);
		vml_vec2 c(max.x, min.y);
		vml_vec2 d(min.x, min.y);
		return contains(circle.center) || Line2D(a, b).intersects(circle) || Line2D(b, c).intersects(circle) || Line2D(c, d).intersects(circle) || Line2D(d, a).intersects(circle);
	}

	bool intersects(const Box2D& aabb) const
	{
		if (aabb.max.x < min.x || aabb.min.x > max.x || aabb.max.y < min.y || aabb.min.y > max.y)
		{
			return false;
		}

		return true;
	}

	bool isIntersected(const Line2D& line) const
	{
		vml_vec2 a(min.x, max.y);
		vml_vec2 b(max.x, max.y);
		vml_vec2 c(max.x, min.y);
		vml_vec2 d(min.x, min.y);
		return contains(line.start) || contains(line.end) || Line2D(a, b).intersects(line) || Line2D(b, c).intersects(line) || Line2D(c, d).intersects(line) || Line2D(d, a).intersects(line);
	}

};

#endif
