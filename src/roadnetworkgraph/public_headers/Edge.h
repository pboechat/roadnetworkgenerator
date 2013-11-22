#ifndef EDGE_H
#define EDGE_H

#include <Defines.h>

namespace RoadNetworkGraph
{

struct Edge
{
	bool highway;
	VertexIndex source;
	VertexIndex destination;

};

}

#endif