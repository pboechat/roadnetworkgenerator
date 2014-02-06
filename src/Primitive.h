#ifndef PRIMITIVE_H
#define PRIMITIVE_H

#pragma once

#include <Constants.h>

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
	volatile int edges[MAX_EDGES_PER_PRIMITIVE];
	vml_vec2 vertices[MAX_VERTICES_PER_PRIMITIVE];
	volatile unsigned int numEdges;
	volatile unsigned int numVertices;

	Primitive() : numEdges(0), numVertices(0) {}

};

//////////////////////////////////////////////////////////////////////////
inline void insertEdge(Primitive& primitive, int edgeIndex)
{
	// FIXME: checking boundaries
	if (primitive.numEdges >= MAX_EDGES_PER_PRIMITIVE)
	{
		throw std::exception("max. number of primitive edges overflow");
	}

	primitive.edges[primitive.numEdges++] = edgeIndex;
}

//////////////////////////////////////////////////////////////////////////
inline void insertVertex(Primitive& primitive, const vml_vec2& vertex)
{
	// FIXME: checking boundaries
	if (primitive.numVertices >= MAX_VERTICES_PER_PRIMITIVE)
	{
		throw std::exception("max. number of primitive vertices overflow");
	}

	primitive.vertices[primitive.numVertices++] = vertex;
}

#endif