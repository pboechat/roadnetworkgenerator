#ifndef QUADRANT_H
#define QUADRANT_H

#pragma once

#include <CpuGpuCompatibility.h>
#include <Box2D.h>

struct Quadrant
{
	unsigned int depth;
	Box2D bounds;
	int edges;

	HOST_AND_DEVICE_CODE Quadrant() : edges(-1) {}
	HOST_AND_DEVICE_CODE ~Quadrant() {}
	
	/*HOST_AND_DEVICE_CODE Quadrant& operator = (const Quadrant& other)
	{
		depth = other.depth;
		bounds = other.bounds;
		edges = other.edges;
		return *this;
	}*/
	
};

#endif