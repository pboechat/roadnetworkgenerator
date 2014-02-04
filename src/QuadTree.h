#ifndef QUADTREE_H
#define QUADTREE_H

#pragma once

#include <CpuGpuCompatibility.h>
#include <Quadrant.h>
#include <QuadrantEdges.h>
#include <Box2D.h>

struct QuadTree
{
	Box2D worldBounds;
	unsigned int maxDepth;
	unsigned int maxQuadrants;
	Quadrant* quadrants;
	QuadrantEdges* quadrantsEdges;
	unsigned int totalNumQuadrants;
	unsigned int numLeafQuadrants;
	volatile int numQuadrantEdges;
#ifdef COLLECT_STATISTICS
	volatile unsigned int numCollisionChecks;
	volatile unsigned int maxEdgesPerQuadrantInUse;
	volatile unsigned int maxResultsPerQueryInUse;
#endif

	HOST_AND_DEVICE_CODE QuadTree() {}
	HOST_AND_DEVICE_CODE ~QuadTree() {}

};

#endif