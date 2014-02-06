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

	HOST_AND_DEVICE_CODE Edge() : attr1(0), attr2(0), attr3(0), attr4(0), numPrimitives(0), owner(-1) {}
	HOST_AND_DEVICE_CODE ~Edge() {}

};


#endif