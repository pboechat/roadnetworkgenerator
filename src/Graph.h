#ifndef GRAPH_H
#define GRAPH_H

#pragma once

#include <Constants.h>
#include <CpuGpuCompatibility.h>
#include <BaseGraph.h>
#include <QuadTree.h>
#include <QueryResults.h>

//////////////////////////////////////////////////////////////////////////
struct Graph : public BaseGraph
{
	unsigned int maxVertices;
	unsigned int maxEdges;
	float snapRadius;
#ifdef _DEBUG
	volatile unsigned long numCollisionChecks;
#endif
#ifdef USE_QUADTREE
	QuadTree* quadtree;
	volatile int lastUsedQueryResults;
	unsigned int maxQueryResults;
	QueryResults* queryResults;
#endif
	// DEBUG:
	volatile int owner;

	HOST_AND_DEVICE_CODE Graph() {}
	HOST_AND_DEVICE_CODE ~Graph() {}

};

#endif