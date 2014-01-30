#ifndef QUADTREE_H
#define QUADTREE_H

#pragma once

#include <CpuGpuCompatibility.h>
#include <Quadrant.h>
#include <QuadrantEdges.h>
#include <Box2D.h>

struct QuadTree
{
	unsigned int maxResultsPerQuery;
	Box2D worldBounds;
	unsigned int maxDepth;
	unsigned int maxQuadrants;
	Quadrant* quadrants;
	QuadrantEdges* quadrantsEdges;
	unsigned int totalNumQuadrants;
	unsigned int numLeafQuadrants;
	int numQuadrantEdges;
#ifdef _DEBUG
	unsigned long numCollisionChecks;
	unsigned int maxEdgesPerQuadrantInUse;
	unsigned int maxResultsPerQueryInUse;
#endif

	HOST_AND_DEVICE_CODE QuadTree() {}
	HOST_AND_DEVICE_CODE ~QuadTree() {}

};

#endif