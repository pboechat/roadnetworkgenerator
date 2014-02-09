#ifndef GRAPH_H
#define GRAPH_H

#pragma once

#include <Constants.h>
#include <CpuGpuCompatibility.h>
#include <BaseGraph.h>
#include <QuadTree.h>

//////////////////////////////////////////////////////////////////////////
struct Graph : public BaseGraph
{
	unsigned int maxVertices;
	unsigned int maxEdges;
	float snapRadius;
#ifdef COLLECT_STATISTICS
	volatile unsigned long numCollisionChecks;
#endif
	QuadTree* quadtree;

	HOST_AND_DEVICE_CODE Graph() {}
	HOST_AND_DEVICE_CODE ~Graph() {}

};

#endif