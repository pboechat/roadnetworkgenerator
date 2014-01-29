#ifndef ROADATTRIBUTES_CUH
#define ROADATTRIBUTES_CUH

#include "Defines.h"

struct RoadAttributes
{
	RoadNetworkGraph::VertexIndex source;
	unsigned int length;
	float angle;

	HOST_AND_DEVICE_CODE RoadAttributes() : source(0), length(0), angle(0) {}
	HOST_AND_DEVICE_CODE RoadAttributes(RoadNetworkGraph::VertexIndex source, unsigned int length, float angle) : source(source), length(length), angle(angle) {}
	HOST_AND_DEVICE_CODE ~RoadAttributes() {}

	HOST_AND_DEVICE_CODE RoadAttributes& operator = (const RoadAttributes& other)
	{
		source = other.source;
		length = other.length;
		angle = other.angle;
		return *this;
	}

};

#endif