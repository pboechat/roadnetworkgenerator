#ifndef ROADNETWORKGRAPH_VERTEX_H
#define ROADNETWORKGRAPH_VERTEX_H

#include <Defines.h>

#include <vector_math.h>

namespace RoadNetworkGraph
{

struct Vertex
{
	VertexIndex index;
	vml_vec2 position;
	EdgeIndex ins[MAX_VERTEX_IN_CONNECTIONS];
	EdgeIndex outs[MAX_VERTEX_OUT_CONNECTIONS];
	unsigned int lastInIndex;
	unsigned int lastOutIndex;
	bool removed;

	Vertex() : removed(false), lastInIndex(0), lastOutIndex(0) {}

};

}

#endif