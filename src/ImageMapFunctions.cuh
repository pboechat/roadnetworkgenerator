#ifndef IMAGEMAPFUNCTIONS_CUH
#define IMAGEMAPFUNCTIONS_CUH

#include <CpuGpuCompatibility.h>
#include <VectorMath.h>
#include <MathExtras.h>

#ifdef PARALLEL
#include <ImageMap.h>
inline DEVICE_CODE unsigned char fetch(const ImageMap* texture, int x, int y)
{
	int x1 = MathExtras::clamp(0, (int)texture->width - 1, x);
	int y1 = MathExtras::clamp(0, (int)texture->height - 1, y);
	return texture->data[(y1 * texture->width) + x1];
}
#define TEX2D(texture, x, y) fetch(texture, x, y)
#else
inline unsigned char fetch(const CudaTexture2DMock& texture, int x, int y)
{
	int x1 = MathExtras::clamp(0, (int)texture.width - 1, x);
	int y1 = MathExtras::clamp(0, (int)texture.height - 1, y);
	return texture.data[(y1 * texture.width) + x1];
}
#define TEX2D(texture, x, y) fetch(texture, x, y)
#endif

#define SCAN(__texture, __origin, __direction, __minDistance, __maxDistance, __greaterSample, __distance) \
	__distance = __maxDistance; \
	__greaterSample = 0; \
	for (int i = __minDistance; i <= __maxDistance; i++) \
	{ \
		vml_vec2 point = __origin + (__direction * (float)i); \
		unsigned char currentSample = TEX2D(__texture, (int)point.x, (int)point.y); \
		if (currentSample > __greaterSample) \
		{ \
			__distance = i; \
			__greaterSample = currentSample; \
		} \
	}

#define CAST_RAY(__texture, __origin, __direction, __length, __threshold, __hit, __hitPoint) \
	__hit = false; \
	for (unsigned int i = 0; i <= __length; i++) \
	{ \
		vml_vec2 point = __origin + (__direction * (float)i); \
		if (TEX2D(__texture, (int)point.x, (int)point.y) > __threshold) \
		{ \
			__hitPoint = point; \
			__hit = true; \
			break; \
		} \
	}

#endif