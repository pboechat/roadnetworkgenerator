#ifndef ROADNETWORKGRAPH_EDGE_CUH
#define ROADNETWORKGRAPH_EDGE_CUH

#include "Defines.h"

namespace RoadNetworkGraph
{

struct Edge
{
	EdgeIndex index;
	VertexIndex source;
	VertexIndex destination;
	unsigned char attr1;
	unsigned char attr2;
	unsigned char attr3;
	unsigned char attr4;

	HOST_AND_DEVICE_CODE Edge() : attr1(0), attr2(0), attr3(0), attr4(0) {}
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

}

#endif