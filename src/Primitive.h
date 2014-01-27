#ifndef ROADNETWORKGRAPH_PRIMITIVE_H
#define ROADNETWORKGRAPH_PRIMITIVE_H

#include "Defines.h"

#include <exception>
#include <vector_math.h>

namespace RoadNetworkGraph
{

//////////////////////////////////////////////////////////////////////////
HOST_CODE enum PrimitiveType
{
	ISOLATED_VERTEX,
	FILAMENT,
	MINIMAL_CYCLE
};

//////////////////////////////////////////////////////////////////////////
HOST_CODE struct Primitive
{
	PrimitiveType type;
	vml_vec2 vertices[MAX_VERTICES_PER_PRIMITIVE];
	unsigned int numVertices;

	Primitive() : numVertices(0) {}

};

//////////////////////////////////////////////////////////////////////////
HOST_CODE inline void insert(Primitive& primitive, const vml_vec2& vertex)
{
	// FIXME: checking boundaries
	if (primitive.numVertices >= MAX_VERTICES_PER_PRIMITIVE)
	{
		THROW_EXCEPTION("max. number of primitive vertices overflow");
	}

	primitive.vertices[primitive.numVertices++] = vertex;
}

}

#endif