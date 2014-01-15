#ifndef MATHEXTRAS_H
#define MATHEXTRAS_H

#include <vector_math.h>
#include <cfloat>

namespace MathExtras
{
const float TWO_PI = 6.28318530717958647692f;
const float PI = 3.14159265358979323846f;
const float HALF_PI = 1.57079632679489661923f;
const float PI_AND_HALF = 4.71238898038468985769f;

template<typename T>
inline static T swap(T& a, T& b)
{
	T& c = a;
	a = b;
	b = c;
}

template<typename T>
inline static T max(T a, T b)
{
	return (a > b) ? a : b;
}

inline static vml_vec2 max(vml_vec2 a, vml_vec2 b)
{
	return vml_vec2(max(a.x, b.x), max(a.y, b.y));
}

inline static vml_vec3 max(vml_vec3 a, vml_vec3 b)
{
	return vml_vec3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}

static vml_vec2 maxPoint(const vml_vec2* points, unsigned int numPoints)
{
	vml_vec2 _max(FLT_MIN, FLT_MIN);
	for (unsigned int i = 0; i < numPoints; i++)
	{
		const vml_vec2& point = points[i];
		_max.x = max(point.x, _max.x);
		_max.y = max(point.y, _max.y);
	}
	return _max;
}

template<typename T>
inline static T min(T a, T b)
{
	return (a < b) ? a : b;
}

inline static vml_vec2 min(vml_vec2 a, vml_vec2 b)
{
	return vml_vec2(min(a.x, b.x), min(a.y, b.y));
}

inline static vml_vec3 min(vml_vec3 a, vml_vec3 b)
{
	return vml_vec3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}

static vml_vec2 minPoint(const vml_vec2* points, unsigned int numPoints)
{
	vml_vec2 _min(FLT_MAX, FLT_MAX);
	for (unsigned int i = 0; i < numPoints; i++)
	{
		const vml_vec2& point = points[i];
		_min.x = min(point.x, _min.x);
		_min.y = min(point.y, _min.y);
	}
	return _min;
}

template<typename T>
inline static T clamp(T min, T max, T x)
{
	return ((x < min) ? min : ((x > max) ? max : x));
}

template<typename T>
inline static vml_vec2 clamp(T min, T max, vml_vec2 a)
{
	return vml_vec2(clamp(min, max, a.x), clamp(min, max, a.y));
}

template<typename T>
inline static vml_vec3 clamp(T min, T max, vml_vec3 a)
{
	return vml_vec3(clamp(min, max, a.x), clamp(min, max, a.y), clamp(min, max, a.z));
}

static float getOrientedAngle(const vml_vec2& a, const vml_vec2& b)
{
	float angle = acos(vml_dot(a, b) / (vml_length(a) * vml_length(b)));
	vml_vec3 e1(a.x, a.y, 0.0f);
	vml_vec3 e2(b.x, b.y, 0.0f);

	if (vml_cross(e1, e2).z > 0)
	{
		return angle;
	}

	else
	{
		return TWO_PI - angle;
	}
}

static void getPolygonInfo(const vml_vec2* vertices, unsigned int numVertices, float& area, vml_vec2& center)
{
	float twiceArea = 0, x = 0, y = 0, f = 0;
	for (unsigned int i = 0, j = numVertices - 1 ; i < numVertices; j = i++) {
		const vml_vec2& p1 = vertices[i]; 
		const vml_vec2& p2 = vertices[j];
		f = p1.x * p2.y - p2.x * p1.y;
		twiceArea += f;
		x += (p1.x + p2.x) * f;
		y += (p1.y + p2.y) * f;
	}
	area = abs(twiceArea * 0.5f);
	f = twiceArea * 3.0f;
	center = vml_vec2(x / f, y / f);
}

}

#endif