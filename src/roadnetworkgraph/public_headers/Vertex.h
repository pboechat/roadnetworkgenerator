#ifndef VERTEX_H
#define VERTEX_H

#include <Defines.h>

#include <glm/glm.hpp>

namespace RoadNetworkGraph
{

struct Vertex
{
	VertexIndex index;
	glm::vec3 position;
	EdgeIndex ins[MAX_VERTEX_CONNECTIONS];
	EdgeIndex outs[MAX_VERTEX_CONNECTIONS];
	unsigned int lastInIndex;
	unsigned int lastOutIndex;
	bool removed;

	Vertex() : removed(false), lastInIndex(0), lastOutIndex(0) {}

};

}

#endif