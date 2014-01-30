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
	unsigned int maxResultsPerQuery;
	float snapRadius;
#ifdef _DEBUG
	unsigned long numCollisionChecks;
#endif
#ifdef USE_QUADTREE
	QuadTree* quadtree;
	int* queryResult;
#endif

	HOST_AND_DEVICE_CODE Graph() {}
	HOST_AND_DEVICE_CODE ~Graph() {}

};

#endif