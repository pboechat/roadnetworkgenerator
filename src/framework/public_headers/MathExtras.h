#ifndef MATHEXTRAS_H
#define MATHEXTRAS_H

#include <vector_math.h>

namespace MathExtras
{
const float TWO_PI = 6.28318530717958647692f;
const float PI = 3.14159265358979323846f;
const float HALF_PI = 1.57079632679489661923f;
const float PI_AND_HALF = 4.71238898038468985769f;

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

}

#endif