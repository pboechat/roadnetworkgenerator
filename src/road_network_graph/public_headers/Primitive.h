#ifndef ROADNETWORKGRAPH_PRIMITIVE_H
#define ROADNETWORKGRAPH_PRIMITIVE_H

#include "Defines.h"

#include <vector_math.h>

namespace RoadNetworkGraph
{

//////////////////////////////////////////////////////////////////////////
enum PrimitiveType
{
	ISOLATED_VERTEX,
	FILAMENT,
	MINIMAL_CYCLE
};

//////////////////////////////////////////////////////////////////////////
struct Primitive
{
	PrimitiveType type;
	vml_vec2 vertices[MAX_VERTICES_PER_PRIMITIVE];
	unsigned int numVertices;

	Primitive() : numVertices(0) {}

};

//////////////////////////////////////////////////////////////////////////
inline void insert(Primitive& primitive, const vml_vec2& vertex)
{
	primitive.vertices[primitive.numVertices++] = vertex;
}

}

#endif