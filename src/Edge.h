#ifndef EDGE_H
#define EDGE_H

#pragma once

#include <CpuGpuCompatibility.h>

struct Edge
{
	int index;
	int source;
	int destination;
	unsigned char attr1;
	unsigned char attr2;
	unsigned char attr3;
	unsigned char attr4;
	unsigned int primitives[2];
	unsigned int numPrimitives;
	volatile int owner;
#ifdef USE_CUDA
	volatile bool readFlag;
#endif

	HOST_AND_DEVICE_CODE Edge() : attr1(0), attr2(0), attr3(0), attr4(0), numPrimitives(0), owner(-1)
#ifdef USE_CUDA
		, readFlag(false) 
#endif
	{}
	HOST_AND_DEVICE_CODE ~Edge() {}

};


#endif