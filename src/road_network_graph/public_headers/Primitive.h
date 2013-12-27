#ifndef ROADNETWORKGRAPH_PRIMITIVE_H
#define ROADNETWORKGRAPH_PRIMITIVE_H

#include <Defines.h>

#include <vector_math.h>

namespace RoadNetworkGraph
{

enum PrimitiveType
{
	ISOLATED_VERTEX,
	FILAMENT,
	CYCLE
};

struct Primitive
{
	PrimitiveType type;
	VertexIndex vertices[MAX_VERTICES_PER_PRIMITIVE];
	unsigned int numVertices;

	Primitive() : numVertices(0) {}

};

}

#endif