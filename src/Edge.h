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
	volatile int owner;

	HOST_AND_DEVICE_CODE Edge() : attr1(0), attr2(0), attr3(0), attr4(0), owner(-1) {}
	HOST_AND_DEVICE_CODE ~Edge() {}
	
	/*HOST_AND_DEVICE_CODE Edge& operator = (const Edge& other)
	{
		index = other.index;
		source = other.source;
		destination = other.destination;
		attr1 = other.attr1;
		attr2 = other.attr2;
		attr3 = other.attr3;
		attr4 = other.attr4;
		return *this;
	}*/

};


#endif