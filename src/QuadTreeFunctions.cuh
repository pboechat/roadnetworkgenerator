#ifndef QUADTREEFUNCTIONS_CUH
#define QUADTREEFUNCTIONS_CUH

#pragma once

#include <QuadTree.h>
#include <Constants.h>
#include <Line2D.h>
#include <Circle2D.h>
#include <VectorMath.h>
#include <QueryResults.h>

//////////////////////////////////////////////////////////////////////////
HOST_AND_DEVICE_CODE void initializeQuadrant(QuadTree* quadtree, const Box2D& quadrantBounds, unsigned int depth = 0, unsigned int index = 0, unsigned int offset = 0, unsigned int levelWidth = 1);
//////////////////////////////////////////////////////////////////////////
HOST_AND_DEVICE_CODE void removeEdgeReferencesInQuadrant(QuadrantEdges* quadrantEdges, int edgeIndex);

//////////////////////////////////////////////////////////////////////////
GLOBAL_CODE void initializeQuadtreeOnDevice(QuadTree* quadtree, Box2D worldBounds, unsigned int depth, unsigned int maxQuadrants, Quadrant* quadrants, QuadrantEdges* quadrantEdges)
{
	quadtree->worldBounds = worldBounds;
	quadtree->maxDepth = depth;
	quadtree->maxQuadrants = maxQuadrants;
	quadtree->quadrants = quadrants;
	quadtree->quadrantsEdges = quadrantEdges;
	quadtree->numQuadrantEdges = 0;
#ifdef _DEBUG
	quadtree->numCollisionChecks = 0;
	quadtree->maxEdgesPerQuadrantInUse = 0;
	quadtree->maxResultsPerQueryInUse = 0;
#endif
	quadtree->totalNumQuadrants = 0;

	for (unsigned int i = 0; i < quadtree->maxDepth; i++)
	{
		unsigned int numQuadrants = (unsigned int)pow(4.0f, (int)i);

		if (i == quadtree->maxDepth - 1)
		{
			quadtree->numLeafQuadrants = numQuadrants;
		}

		quadtree->totalNumQuadrants += numQuadrants;
	}

	initializeQuadrant(quadtree, worldBounds);
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE void initializeQuadtreeOnHost(QuadTree* quadtree, Box2D worldBounds, unsigned int depth, unsigned int maxQuadrants, Quadrant* quadrants, QuadrantEdges* quadrantEdges)
{
	quadtree->worldBounds = worldBounds;
	quadtree->maxDepth = depth;
	quadtree->maxQuadrants = maxQuadrants;
	quadtree->quadrants = quadrants;
	quadtree->quadrantsEdges = quadrantEdges;
	quadtree->numQuadrantEdges = 0;
#ifdef _DEBUG
	quadtree->numCollisionChecks = 0;
	quadtree->maxEdgesPerQuadrantInUse = 0;
	quadtree->maxResultsPerQueryInUse = 0;
#endif
	quadtree->totalNumQuadrants = 0;

	for (unsigned int i = 0; i < quadtree->maxDepth; i++)
	{
		unsigned int numQuadrants = (unsigned int)pow(4.0f, (int)i);

		if (i == quadtree->maxDepth - 1)
		{
			quadtree->numLeafQuadrants = numQuadrants;
		}

		quadtree->totalNumQuadrants += numQuadrants;
	}

	initializeQuadrant(quadtree, worldBounds);
}

//////////////////////////////////////////////////////////////////////////
GLOBAL_CODE void updateNonPointerFields(QuadTree* quadtree, int numQuadrantEdges, Box2D worldBounds, unsigned int maxDepth, unsigned int maxQuadrants, unsigned int totalNumQuadrants, unsigned int numLeafQuadrants
#ifdef _DEBUG
	, unsigned long numCollisionChecks
	, unsigned int maxEdgesPerQuadrantInUse
	, unsigned int maxResultsPerQueryInUse
#endif
)
{
	quadtree->numQuadrantEdges = numQuadrantEdges;
	quadtree->worldBounds = worldBounds;
	quadtree->maxQuadrants = maxQuadrants;
	quadtree->totalNumQuadrants = totalNumQuadrants;
	quadtree->numLeafQuadrants = numLeafQuadrants;
#ifdef _DEBUG
	quadtree->numCollisionChecks = numCollisionChecks;
	quadtree->maxEdgesPerQuadrantInUse = maxEdgesPerQuadrantInUse;
	quadtree->maxResultsPerQueryInUse = maxResultsPerQueryInUse;
#endif
}

//////////////////////////////////////////////////////////////////////////
HOST_AND_DEVICE_CODE void initializeQuadrant(QuadTree* quadtree, const Box2D& quadrantBounds, unsigned int depth, unsigned int index, unsigned int offset, unsigned int levelWidth)
{
	unsigned int i = offset + index;

	// FIXME: checking boundaries
	if (i >= quadtree->maxQuadrants)
	{
		THROW_EXCEPTION("max. quadrants overflow");
	}

	Quadrant* quadrant = &quadtree->quadrants[i];

	quadrant->depth = depth;
	quadrant->bounds = quadrantBounds;

	if (depth == quadtree->maxDepth - 1) // leaf
	{
		int quadrantEdgesIndex = ATOMIC_ADD(quadtree->numQuadrantEdges, int, 1);
		QuadrantEdges* quadrantEdges = &quadtree->quadrantsEdges[quadrantEdgesIndex];
		quadrant->edges = quadrantEdgesIndex;
		return;
	}

	unsigned int baseIndex = (index * 4);
	unsigned int newOffset = offset + levelWidth;
	unsigned int newLevelWidth = levelWidth * 4;
	unsigned int newDepth = depth + 1;
	vml_vec2 subQuadrantSize = quadrantBounds.getExtents() / 2.0f;

	for (unsigned int y = 0, i = 0; y < 2; y++)
	{
		vml_vec2 quadrantBoundsMin = quadrantBounds.getMin();
		float subQuadrantY = quadrantBoundsMin.y + ((float)y * subQuadrantSize.y);

		for (unsigned int x = 0; x < 2; x++, i++)
		{
			initializeQuadrant(quadtree, Box2D(quadrantBoundsMin.x + ((float)x * subQuadrantSize.x), subQuadrantY, subQuadrantSize.x, subQuadrantSize.y), newDepth, baseIndex + i, newOffset, newLevelWidth);
		}
	}
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void insert(QuadTree* quadtree, int edgeIndex, const Line2D& edgeLine, unsigned int index = 0, unsigned int offset = 0, unsigned int levelWidth = 1)
{
	Quadrant* quadrant = &quadtree->quadrants[offset + index];

	if (quadrant->bounds.isIntersected(edgeLine))
	{
		if (quadrant->depth == quadtree->maxDepth - 1)
		{
			// FIXME: checking invariants
			if (quadrant->edges == -1)
			{
				THROW_EXCEPTION("quadrant.edges == -1");
			}

			QuadrantEdges* quadrantEdges = &quadtree->quadrantsEdges[quadrant->edges];

			// FIXME: checking boundaries
			if (quadrantEdges->lastEdgeIndex >= MAX_EDGES_PER_QUADRANT)
			{
				THROW_EXCEPTION("max. edges per quadrant overflow");
			}

			quadrantEdges->edges[quadrantEdges->lastEdgeIndex++] = edgeIndex;

#ifdef _DEBUG
			// URGENT: protect from concurrent writes!!!
			quadtree->maxEdgesPerQuadrantInUse = MathExtras::max(quadtree->maxEdgesPerQuadrantInUse, quadrantEdges->lastEdgeIndex);
#endif
		}

		else
		{
			unsigned int baseIndex = (index * 4);
			unsigned int newOffset = offset + levelWidth;
			unsigned int newLevelWidth = levelWidth * 4;
			insert(quadtree, edgeIndex, edgeLine, baseIndex, newOffset, newLevelWidth);
			insert(quadtree, edgeIndex, edgeLine, baseIndex + 1, newOffset, newLevelWidth);
			insert(quadtree, edgeIndex, edgeLine, baseIndex + 2, newOffset, newLevelWidth);
			insert(quadtree, edgeIndex, edgeLine, baseIndex + 3, newOffset, newLevelWidth);
		}
	}

#ifdef _DEBUG
	ATOMIC_ADD(quadtree->numCollisionChecks, unsigned int, 1);
#endif
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void remove(QuadTree* quadtree, int edgeIndex, const Line2D& edgeLine, unsigned int index = 0, unsigned int offset = 0, unsigned int levelWidth = 1)
{
	Quadrant* quadrant = &quadtree->quadrants[offset + index];

	if (quadrant->bounds.isIntersected(edgeLine))
	{
		if (quadrant->depth == quadtree->maxDepth - 1)
		{
			// FIXME: checking invariants
			if (quadrant->edges == -1)
			{
				THROW_EXCEPTION("quadrant.edges == -1");
			}

			QuadrantEdges* quadrantEdges = &quadtree->quadrantsEdges[quadrant->edges];
			removeEdgeReferencesInQuadrant(quadrantEdges, edgeIndex);
		}

		else
		{
			unsigned int baseIndex = (index * 4);
			unsigned int newOffset = offset + levelWidth;
			unsigned int newLevelWidth = levelWidth * 4;
			insert(quadtree, edgeIndex, edgeLine, baseIndex, newOffset, newLevelWidth);
			insert(quadtree, edgeIndex, edgeLine, baseIndex + 1, newOffset, newLevelWidth);
			insert(quadtree, edgeIndex, edgeLine, baseIndex + 2, newOffset, newLevelWidth);
			insert(quadtree, edgeIndex, edgeLine, baseIndex + 3, newOffset, newLevelWidth);
		}
	}

#ifdef _DEBUG
	ATOMIC_ADD(quadtree->numCollisionChecks, unsigned int, 1);
#endif
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void removeEdgeReferencesInQuadrant(QuadrantEdges* quadrantEdges, int edgeIndex)
{
	// FIXME: checking invariants
	if (quadrantEdges == 0)
	{
		THROW_EXCEPTION("quadrantEdges == 0");
	}

	// FIXME: checking boundaries
	if (quadrantEdges->lastEdgeIndex == 0)
	{
		THROW_EXCEPTION("tried to remove edge from an empty quadrant");
	}

	unsigned int i = 0;
	bool found = false;

	for (unsigned int j = 0; j < quadrantEdges->lastEdgeIndex; j++)
	{
		if (quadrantEdges->edges[j] == edgeIndex)
		{
			i = j;
			found = true;
			break;
		}
	}

	// FIXME: checking invariants
	if (!found)
	{
		int a = 0;
		THROW_EXCEPTION("!found");
	}

	for (unsigned int j = i; j < quadrantEdges->lastEdgeIndex - 1; j++)
	{
		quadrantEdges->edges[j] = quadrantEdges->edges[j + 1];
	}

	quadrantEdges->lastEdgeIndex--;
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
DEVICE_CODE void query(QuadTree* quadtree, const T& shape, QueryResults* queryResult, unsigned int offset = 0)
{
	queryResult->numResults = offset;
	recursiveQuery(quadtree, shape, queryResult, 0, 0, 1);
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
DEVICE_CODE void recursiveQuery(QuadTree* quadtree, const T& shape, QueryResults* queryResults, unsigned int index, unsigned int offset, unsigned int levelWidth)
{
	int quadrantIndex = offset + index;

	// FIXME: checking invariants
	if (quadrantIndex >= (int)quadtree->maxQuadrants)
	{
		THROW_EXCEPTION("quadrantIndex >= quadtree->maxQuadrants");
	}

	Quadrant* quadrant = &quadtree->quadrants[quadrantIndex];

	if (quadrant->bounds.intersects(shape))
	{
		if (quadrant->depth == quadtree->maxDepth - 1)
		{
			// FIXME: checking invariants
			if (quadrant->edges == -1)
			{
				THROW_EXCEPTION("quadrant.edges == -1");
			}

			QuadrantEdges* quadrantEdges = &quadtree->quadrantsEdges[quadrant->edges];

			for (unsigned int i = 0; i < quadrantEdges->lastEdgeIndex; i++)
			{
				// FIXME: checking boundaries
				if (queryResults->numResults >= MAX_RESULTS_PER_QUERY)
				{
					THROW_EXCEPTION("max. results per query overflow");
				}

				queryResults->results[queryResults->numResults++] = quadrantEdges->edges[i];
			}

#ifdef _DEBUG
			// URGENT: protect from concurrent writes!!!
			quadtree->maxResultsPerQueryInUse = MathExtras::max(quadtree->maxResultsPerQueryInUse, queryResults->numResults);
#endif
		}

		else
		{
			unsigned int baseIndex = (index * 4);
			unsigned int newOffset = offset + levelWidth;
			unsigned int newLevelWidth = levelWidth * 4;
			recursiveQuery(quadtree, shape, queryResults, baseIndex, newOffset, newLevelWidth);
			recursiveQuery(quadtree, shape, queryResults, baseIndex + 1, newOffset, newLevelWidth);
			recursiveQuery(quadtree, shape, queryResults, baseIndex + 2, newOffset, newLevelWidth);
			recursiveQuery(quadtree, shape, queryResults, baseIndex + 3, newOffset, newLevelWidth);
		}
	}

#ifdef _DEBUG
	ATOMIC_ADD(quadtree->numCollisionChecks, unsigned int, 1);
#endif
}

#ifdef _DEBUG
//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getAllocatedMemory(QuadTree* quadtree)
{
	unsigned int quadrantsBufferMemory = quadtree->totalNumQuadrants * sizeof(Quadrant);
	unsigned int quadrantsEdgesBufferMemory = quadtree->numLeafQuadrants * sizeof(QuadrantEdges);
	return (quadrantsBufferMemory + quadrantsEdgesBufferMemory);
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getMemoryInUse(QuadTree* quadtree)
{
	unsigned int quadrantsBufferMemoryInUse = quadtree->totalNumQuadrants * sizeof(Quadrant);
	unsigned int quadrantsEdgesBufferMemoryInUse = 0;

	for (unsigned int i = 0; i < quadtree->numLeafQuadrants; i++)
	{
		QuadrantEdges& quadrantEdges = quadtree->quadrantsEdges[i];
		quadrantsEdgesBufferMemoryInUse += sizeof(int) * quadrantEdges.lastEdgeIndex + sizeof(unsigned int);
	}

	return (quadrantsBufferMemoryInUse + quadrantsEdgesBufferMemoryInUse);
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned long getNumCollisionChecks(QuadTree* quadtree)
{
	return quadtree->numCollisionChecks;
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getMaxEdgesPerQuadrantInUse(QuadTree* quadtree)
{
	return quadtree->maxEdgesPerQuadrantInUse;
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getMaxResultsPerQueryInUse(QuadTree* quadtree)
{
	return quadtree->maxResultsPerQueryInUse;
}
#endif


#endif