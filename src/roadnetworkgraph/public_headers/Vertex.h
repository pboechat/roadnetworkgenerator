#ifndef VERTEX_H
#define VERTEX_H

#include <Defines.h>

#include <glm/glm.hpp>

namespace RoadNetworkGraph
{

struct Vertex
{
	VertexIndex index;
	VertexIndex source;
	glm::vec3 position;
	EdgeIndex connections[MAX_VERTEX_CONNECTIONS];
	unsigned int lastConnectionIndex;

	Vertex() : lastConnectionIndex(0) {}

};

}

#endif