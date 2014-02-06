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
	volatile int ins[MAX_VERTEX_IN_CONNECTIONS];
	volatile int outs[MAX_VERTEX_OUT_CONNECTIONS];
	volatile int adjacencies[MAX_VERTEX_IN_CONNECTIONS + MAX_VERTEX_OUT_CONNECTIONS];
	volatile unsigned int numIns;
	volatile unsigned int numOuts;
	volatile unsigned int numAdjacencies;
	bool removed;

	HOST_AND_DEVICE_CODE Vertex() : index(-1), numIns(0), numOuts(0), numAdjacencies(0), removed(false) {}
	HOST_AND_DEVICE_CODE ~Vertex() {}

};

#endif