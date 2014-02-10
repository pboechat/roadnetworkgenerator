#ifndef COLLISION_H
#define COLLISION_H

#include <CpuGpuCompatibility.h>
#include <VectorMath.h>

struct Collision
{
	int edge1;
	int edge2;
	vec2FieldDeclaration(Intersection, HOST_AND_DEVICE_CODE)

	HOST_AND_DEVICE_CODE Collision() : edge1(-1), edge2(-1) {}
	HOST_AND_DEVICE_CODE ~Collision() {}

};

#endif