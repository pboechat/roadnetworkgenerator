#ifndef BASEGRAPH_H
#define BASEGRAPH_H

#pragma once

#include <CpuGpuCompatibility.h>
#include <Vertex.h>
#include <Edge.h>

struct BaseGraph
{
	int numVertices;
	int numEdges;
	Vertex* vertices;
	Edge* edges;

	HOST_AND_DEVICE_CODE BaseGraph() {}
	HOST_AND_DEVICE_CODE ~BaseGraph() {}

};

#endif