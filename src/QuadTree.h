#ifndef ROADNETWORKGRAPH_QUADREE_H
#define ROADNETWORKGRAPH_QUADREE_H

#include "Defines.h"
#include <Quadrant.h>
#include <QuadrantEdges.h>

#include <Line2D.h>
#include <Circle2D.h>
#include <Box2D.h>

#include <vector_math.h>

#include <cmath>
#include <exception>

namespace RoadNetworkGraph
{

HOST_AND_DEVICE_CODE struct QuadTree
{
	unsigned int maxResultsPerQuery;
	Box2D worldBounds;
	unsigned int maxDepth;
	unsigned int maxQuadrants;
	Quadrant* quadrants;
	QuadrantEdges* quadrantsEdges;
	unsigned int totalNumQuadrants;
	unsigned int numLeafQuadrants;
	QuadrantEdgesIndex numQuadrantEdges;
#ifdef _DEBUG
	unsigned long numCollisionChecks;
	unsigned int maxEdgesPerQuadrantInUse;
	unsigned int maxResultsPerQueryInUse;
#endif

};

//////////////////////////////////////////////////////////////////////////
GLOBAL_CODE void initializeQuadtree(QuadTree* quadtree, const Box2D& worldBounds, unsigned int depth, unsigned int maxResultsPerQuery, unsigned int maxQuadrants, Quadrant* quadrants, QuadrantEdges* quadrantEdges);
//////////////////////////////////////////////////////////////////////////
HOST_AND_DEVICE_CODE void insert(QuadTree* quadtree, EdgeIndex edgeIndex, const Line2D& edgeLine, unsigned int index = 0, unsigned int offset = 0, unsigned int levelWidth = 1);
//////////////////////////////////////////////////////////////////////////
HOST_AND_DEVICE_CODE void remove(QuadTree* quadtree, EdgeIndex edgeIndex, const Line2D& edgeLine, unsigned int index = 0, unsigned int offset = 0, unsigned int levelWidth = 1);
//////////////////////////////////////////////////////////////////////////
template<typename T>
HOST_AND_DEVICE_CODE void query(QuadTree* quadtree, const T& shape, EdgeIndex* queryResult, unsigned int& size, unsigned int offset = 0)
{
	size = offset;
	recursiveQuery(quadtree, shape, queryResult, size, 0, 0, 1);
}
//////////////////////////////////////////////////////////////////////////
template<typename T>
HOST_AND_DEVICE_CODE void recursiveQuery(QuadTree* quadtree, const T& shape, EdgeIndex* queryResult, unsigned int& size, unsigned int index, unsigned int offset, unsigned int levelWidth)
{
	QuadrantIndex quadrantIndex = offset + index;

	// FIXME: checking invariants
	if (quadrantIndex >= (int)quadtree->maxQuadrants)
	{
		THROW_EXCEPTION("quadrantIndex >= quadtree->maxQuadrants");
	}

	Quadrant& quadrant = quadtree->quadrants[quadrantIndex];

	if (quadrant.bounds.intersects(shape))
	{
		if (quadrant.depth == quadtree->maxDepth - 1)
		{
			// FIXME: checking invariants
			if (quadrant.edges == -1)
			{
				THROW_EXCEPTION("quadrant.edges == -1");
			}

			QuadrantEdges& quadrantEdges = quadtree->quadrantsEdges[quadrant.edges];

			for (unsigned int i = 0; i < quadrantEdges.lastEdgeIndex; i++)
			{
				queryResult[size++] = quadrantEdges.edges[i];

				// FIXME: checking boundaries
				if (size >= quadtree->maxResultsPerQuery)
				{
					THROW_EXCEPTION("max. results per query overflow");
				}
			}

#ifdef _DEBUG
			quadtree->maxResultsPerQueryInUse = MathExtras::max(quadtree->maxResultsPerQueryInUse, size);
#endif
		}

		else
		{
			unsigned int baseIndex = (index * 4);
			unsigned int newOffset = offset + levelWidth;
			unsigned int newLevelWidth = levelWidth * 4;
			recursiveQuery(quadtree, shape, queryResult, size, baseIndex, newOffset, newLevelWidth);
			recursiveQuery(quadtree, shape, queryResult, size, baseIndex + 1, newOffset, newLevelWidth);
			recursiveQuery(quadtree, shape, queryResult, size, baseIndex + 2, newOffset, newLevelWidth);
			recursiveQuery(quadtree, shape, queryResult, size, baseIndex + 3, newOffset, newLevelWidth);
		}
	}

#ifdef _DEBUG
	quadtree->numCollisionChecks++;
#endif
}
#ifdef _DEBUG
//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getAllocatedMemory(QuadTree* quadtree);
//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getMemoryInUse(QuadTree* quadtree);
//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned long getNumCollisionChecks(QuadTree* quadtree);
//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getMaxEdgesPerQuadrantInUse(QuadTree* quadtree);
//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getMaxResultsPerQueryInUse(QuadTree* quadtree);
#endif

}

#endif