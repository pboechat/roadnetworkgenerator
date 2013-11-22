#ifndef ROADATTRIBUTES_H
#define ROADATTRIBUTES_H

#include <Graph.h>

struct RoadAttributes
{
	RoadNetworkGraph::VertexIndex source;
	unsigned int length;
	float angle;
	bool highway;

	RoadAttributes() {}
	RoadAttributes(RoadNetworkGraph::VertexIndex source, unsigned int length, float angle, bool highway) : source(source), length(length), angle(angle), highway(highway) {}
	~RoadAttributes() {}

	RoadAttributes& operator =(const RoadAttributes& other)
	{
		source = other.source;
		length = other.length;
		angle = other.angle;
		highway = other.highway;
		return *this;
	}

};

#endif