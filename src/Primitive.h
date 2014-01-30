#ifndef PRIMITIVE_H
#define PRIMITIVE_H

#pragma once

#include <Constants.h>
#include <VectorMath.h>

#include <exception>

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
	// FIXME: checking boundaries
	if (primitive.numVertices >= MAX_VERTICES_PER_PRIMITIVE)
	{
		throw std::exception("max. number of primitive vertices overflow");
	}

	primitive.vertices[primitive.numVertices++] = vertex;
}

#endif