#ifndef ROADNETWORKGRAPH_EDGE_H
#define ROADNETWORKGRAPH_EDGE_H

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

	Edge() : attr1(0), attr2(0), attr3(0), attr4(0)
	{
	}

};

}

#endif