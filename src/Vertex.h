#ifndef VERTEX_H
#define VERTEX_H

#pragma once

#include <Constants.h>
#include <CpuGpuCompatibility.h>
#include <VectorMath.h>

//////////////////////////////////////////////////////////////////////////
struct Vertex
{
	int index;
	vec2FieldDeclaration(Position, HOST_AND_DEVICE_CODE)
	int ins[MAX_VERTEX_IN_CONNECTIONS];
	int outs[MAX_VERTEX_OUT_CONNECTIONS];
	int adjacencies[MAX_VERTEX_IN_CONNECTIONS + MAX_VERTEX_OUT_CONNECTIONS];
	unsigned int numIns;
	unsigned int numOuts;
	unsigned int numAdjacencies;
	bool removed;

	HOST_AND_DEVICE_CODE Vertex() : removed(false), numIns(0), numOuts(0), numAdjacencies(0) {}
	HOST_AND_DEVICE_CODE ~Vertex() {}
	
	/*HOST_AND_DEVICE_CODE Vertex& operator = (const Vertex& other)
	{
		index = other.index;
		setPosition(other.getPosition());
		ins = other.ins;
		outs = other.outs;
		adjacencies = other.adjacencies;
		numIns = other.numIns;
		numOuts = other.numOuts;
		removed = other.removed;
		return *this;
	}*/

};

#endif