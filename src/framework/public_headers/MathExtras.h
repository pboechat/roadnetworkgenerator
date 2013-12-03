#ifndef MATHF_H
#define MATHF_H

#include <glm/glm.hpp>

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

template<typename T>
inline static T min(T a, T b)
{
	return (a < b) ? a : b;
}

template<typename T>
inline static T clamp(T min, T max, T x)
{
	return ((x < min) ? min : ((x > max) ? max : x));
}

static float getOrientedAngle(const glm::vec3& a, const glm::vec3& b)
{
	float angle = acos(glm::dot(a, b) / (glm::length(a) * glm::length(b)));
	if (glm::cross(a, b).z > 0)
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