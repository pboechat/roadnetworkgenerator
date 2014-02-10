#ifndef MATHEXTRAS_H
#define MATHEXTRAS_H

#pragma once

#include <CpuGpuCompatibility.h>
#include <VectorMath.h>

#include <cfloat>

#ifdef min
#undef min
#endif

#ifdef max
#undef max
#endif

namespace MathExtras
{

#define TWO_PI 6.28318530717958647692f
#define PI 3.14159265358979323846f
#define HALF_PI 1.57079632679489661923f
#define PI_AND_HALF 4.71238898038468985769f
#define EPSILON 0.00001f

inline HOST_AND_DEVICE_CODE bool isZero(float a)
{
	return a >= -EPSILON && a <= EPSILON;
}

template<typename T>
inline HOST_AND_DEVICE_CODE T acos(T a)
{
	return ::acos(a);
}

template<typename T>
inline HOST_AND_DEVICE_CODE T swap(T& a, T& b)
{
	T& c = a;
	a = b;
	b = c;
}

template<typename T>
inline HOST_AND_DEVICE_CODE T max(T a, T b)
{
	return (a > b) ? a : b;
}

inline HOST_AND_DEVICE_CODE vml_vec2 max(vml_vec2 a, vml_vec2 b)
{
	return vml_vec2(max(a.x, b.x), max(a.y, b.y));
}

inline HOST_AND_DEVICE_CODE vml_vec3 max(vml_vec3 a, vml_vec3 b)
{
	return vml_vec3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}

inline HOST_AND_DEVICE_CODE vml_vec2 maxPoint(const vml_vec2* points, unsigned int numPoints)
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
inline HOST_AND_DEVICE_CODE T min(T a, T b)
{
	return (a < b) ? a : b;
}

template<typename T>
inline HOST_AND_DEVICE_CODE T pow(T a, T b)
{
	return (T)::pow((double)a, (double)b);
}

inline HOST_AND_DEVICE_CODE vml_vec2 min(vml_vec2 a, vml_vec2 b)
{
	return vml_vec2(min(a.x, b.x), min(a.y, b.y));
}

inline HOST_AND_DEVICE_CODE vml_vec3 min(vml_vec3 a, vml_vec3 b)
{
	return vml_vec3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}

inline HOST_AND_DEVICE_CODE vml_vec2 minPoint(const vml_vec2* points, unsigned int numPoints)
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
inline HOST_AND_DEVICE_CODE T clamp(T min, T max, T x)
{
	return ((x < min) ? min : ((x > max) ? max : x));
}

template<typename T>
inline HOST_AND_DEVICE_CODE vml_vec2 clamp(T min, T max, vml_vec2 a)
{
	return vml_vec2(clamp(min, max, a.x), clamp(min, max, a.y));
}

template<typename T>
inline HOST_AND_DEVICE_CODE vml_vec3 clamp(T min, T max, vml_vec3 a)
{
	return vml_vec3(clamp(min, max, a.x), clamp(min, max, a.y), clamp(min, max, a.z));
}

inline HOST_AND_DEVICE_CODE float getAngle(const vml_vec2& a, const vml_vec2& b)
{
	float angle = acos(vml_dot(a, b) / (vml_length(a) * vml_length(b)));
	vml_vec3 e1(a.x, a.y, 0.0f);
	vml_vec3 e2(b.x, b.y, 0.0f);

	if (vml_cross(e1, e2).z >= 0)
	{
		return angle;
	}

	else
	{
		return TWO_PI - angle;
	}
}

inline HOST_AND_DEVICE_CODE float getOrientedAngle(const vml_vec2& a, const vml_vec2& b)
{
	float angle = acos(vml_dot(a, b) / (vml_length(a) * vml_length(b)));
	vml_vec3 e1(a.x, a.y, 0.0f);
	vml_vec3 e2(b.x, b.y, 0.0f);

	if (vml_cross(e1, e2).z >= 0)
	{
		return angle;
	}

	else
	{
		return -angle;
	}
}

}

#endif