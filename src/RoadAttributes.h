#ifndef ROADATTRIBUTES_H
#define ROADATTRIBUTES_H

#include "Defines.h"
#include <Graph.h>

HOST_AND_DEVICE_CODE struct RoadAttributes
{
	RoadNetworkGraph::VertexIndex source;
	unsigned int length;
	float angle;

	RoadAttributes() {}
	RoadAttributes(RoadNetworkGraph::VertexIndex source, unsigned int length, float angle) : source(source), length(length), angle(angle) {}

};

#endif