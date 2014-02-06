#ifndef QUADTREEFUNCTIONS_CUH
#define QUADTREEFUNCTIONS_CUH

#pragma once

#include <QuadTree.h>
#include <Constants.h>
#include <Line2D.h>
#include <Circle2D.h>
#include <VectorMath.h>
#include <QueryResults.h>
#include <QuadTreeStacks.h>

//////////////////////////////////////////////////////////////////////////
HOST_AND_DEVICE_CODE void removeEdgeReferencesInQuadrant(QuadrantEdges& quadrantEdges, int edgeIndex);

//////////////////////////////////////////////////////////////////////////
HOST_AND_DEVICE_CODE void initializeQuadtreeOnHost(QuadTree* quadtree, Box2D worldBounds, unsigned int depth, unsigned int maxQuadrants, Quadrant* quadrants, QuadrantEdges* quadrantEdges)
{
	quadtree->worldBounds = worldBounds;
	quadtree->maxDepth = depth;
	quadtree->maxQuadrants = maxQuadrants;
	quadtree->quadrants = quadrants;
	quadtree->quadrantsEdges = quadrantEdges;
	quadtree->numQuadrantEdges = 0;
#ifdef COLLECT_STATISTICS
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

	unsigned int index, offset, levelWidth;
	Box2D quadrantBounds;

	InitializationQuadTreeStack stack;

	stack.push(0, 0, 1, 0, worldBounds);

	while (stack.notEmpty())
	{
		stack.pop(index, offset, levelWidth, depth, quadrantBounds);

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
			int i = ATOMIC_ADD(quadtree->numQuadrantEdges, int, 1);
			QuadrantEdges* quadrantEdges = &quadtree->quadrantsEdges[i];
			quadrant->edges = i;
		}
		else
		{
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
					Box2D subQuadrantBounds(quadrantBoundsMin.x + ((float)x * subQuadrantSize.x), subQuadrantY, subQuadrantSize.x, subQuadrantSize.y);
					stack.push(baseIndex + i, newOffset, newLevelWidth, newDepth, subQuadrantBounds);
				}
			}
		}
	}
}

//////////////////////////////////////////////////////////////////////////
GLOBAL_CODE void initializeQuadtreeOnDevice(QuadTree* quadtree, Box2D worldBounds, unsigned int depth, unsigned int maxQuadrants, Quadrant* quadrants, QuadrantEdges* quadrantEdges)
{
	initializeQuadtreeOnHost(quadtree, worldBounds, depth, maxQuadrants, quadrants, quadrantEdges);
}

//////////////////////////////////////////////////////////////////////////
GLOBAL_CODE void updateNonPointerFields(QuadTree* quadtree, int numQuadrantEdges, Box2D worldBounds, unsigned int maxDepth, unsigned int maxQuadrants, unsigned int totalNumQuadrants, unsigned int numLeafQuadrants
#ifdef COLLECT_STATISTICS
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
#ifdef COLLECT_STATISTICS
	quadtree->numCollisionChecks = numCollisionChecks;
	quadtree->maxEdgesPerQuadrantInUse = maxEdgesPerQuadrantInUse;
	quadtree->maxResultsPerQueryInUse = maxResultsPerQueryInUse;
#endif
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void insert(QuadTree* quadtree, int edgeIndex, const Line2D& edgeLine)
{
	unsigned int index, offset, levelWidth;

	SimpleQuadTreeStack stack;

	stack.push(0, 0, 1);

	while (stack.notEmpty())
	{
		stack.pop(index, offset, levelWidth);

		Quadrant& quadrant = quadtree->quadrants[index + offset];

		if (quadrant.bounds.isIntersected(edgeLine))
		{
			if (quadrant.depth == quadtree->maxDepth - 1)
			{
				// FIXME: checking invariants
				if (quadrant.edges == -1)
				{
					THROW_EXCEPTION("quadrant.edges == -1");
				}

				QuadrantEdges* quadrantEdges = &quadtree->quadrantsEdges[quadrant.edges];

				// FIXME: checking boundaries
				if (quadrantEdges->lastEdgeIndex >= MAX_EDGES_PER_QUADRANT)
				{
					THROW_EXCEPTION("max. edges per quadrant overflow");
				}

				unsigned int i = ATOMIC_ADD(quadrantEdges->lastEdgeIndex, unsigned int, 1);
				quadrantEdges->edges[i] = edgeIndex;

#ifdef COLLECT_STATISTICS
				ATOMIC_MAX(quadtree->maxEdgesPerQuadrantInUse, unsigned int, quadrantEdges->lastEdgeIndex);
#endif
			}

			else
			{
				unsigned int baseIndex = (index * 4);
				unsigned int newOffset = offset + levelWidth;
				unsigned int newLevelWidth = levelWidth * 4;

				stack.push(baseIndex, newOffset, newLevelWidth);
				stack.push(baseIndex + 1, newOffset, newLevelWidth);
				stack.push(baseIndex + 2, newOffset, newLevelWidth);
				stack.push(baseIndex + 3, newOffset, newLevelWidth);
			}
		}
#ifdef COLLECT_STATISTICS
	ATOMIC_ADD(quadtree->numCollisionChecks, unsigned int, 1);
#endif
	}
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void remove(QuadTree* quadtree, int edgeIndex, const Line2D& edgeLine)
{
	unsigned int index, offset, levelWidth;

	SimpleQuadTreeStack stack;

	stack.push(0, 0, 1);

	while (stack.notEmpty())
	{
		stack.pop(index, offset, levelWidth);

		Quadrant& quadrant = quadtree->quadrants[index + offset];

		if (quadrant.bounds.isIntersected(edgeLine))
		{
			if (quadrant.depth == quadtree->maxDepth - 1)
			{
				// FIXME: checking invariants
				if (quadrant.edges == -1)
				{
					THROW_EXCEPTION("quadrant.edges == -1");
				}

				QuadrantEdges& quadrantEdges = quadtree->quadrantsEdges[quadrant.edges];
				removeEdgeReferencesInQuadrant(quadrantEdges, edgeIndex);
			}

			else
			{
				unsigned int baseIndex = (index * 4);
				unsigned int newOffset = offset + levelWidth;
				unsigned int newLevelWidth = levelWidth * 4;
				
				stack.push(baseIndex, newOffset, newLevelWidth);
				stack.push(baseIndex + 1, newOffset, newLevelWidth);
				stack.push(baseIndex + 2, newOffset, newLevelWidth);
				stack.push(baseIndex + 3, newOffset, newLevelWidth);
			}
		}

#ifdef COLLECT_STATISTICS
		ATOMIC_ADD(quadtree->numCollisionChecks, unsigned int, 1);
#endif
	}
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void removeEdgeReferencesInQuadrant(QuadrantEdges& quadrantEdges, int edgeIndex)
{
	// FIXME: checking boundaries
	if (quadrantEdges.lastEdgeIndex == 0)
	{
		THROW_EXCEPTION("tried to remove edge from an empty quadrant");
	}

	unsigned int i = 0;
	bool found = false;

	for (unsigned int j = 0; j < quadrantEdges.lastEdgeIndex; j++)
	{
		if (quadrantEdges.edges[j] == edgeIndex)
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

	for (unsigned int j = i; j < quadrantEdges.lastEdgeIndex - 1; j++)
	{
		quadrantEdges.edges[j] = quadrantEdges.edges[j + 1];
	}

	quadrantEdges.lastEdgeIndex--;
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void query(QuadTree* quadtree, const Line2D& edgeLine, QueryResults& queryResults)
{
	unsigned int index, offset, levelWidth;

	queryResults.numResults = 0;

	SimpleQuadTreeStack stack;

	stack.push(0, 0, 1);

	while (stack.notEmpty())
	{
		stack.pop(index, offset, levelWidth);

		int quadrantIndex = index + offset;

		// FIXME: checking invariants
		if (quadrantIndex >= (int)quadtree->maxQuadrants)
		{
			THROW_EXCEPTION("quadrantIndex >= quadtree->maxQuadrants");
		}

		Quadrant* quadrant = &quadtree->quadrants[quadrantIndex];

		if (quadrant->bounds.isIntersected(edgeLine))
		{
			if (quadrant->depth == quadtree->maxDepth - 1)
			{
				// FIXME: checking invariants
				if (quadrant->edges == -1)
				{
					THROW_EXCEPTION("quadrant.edges == -1");
				}

				// FIXME: checking boundaries
				if (queryResults.numResults >= MAX_RESULTS_PER_QUERY)
				{
					THROW_EXCEPTION("max. results per query overflow");
				}

				queryResults.results[queryResults.numResults++] = quadrant->edges;

	#ifdef COLLECT_STATISTICS
				ATOMIC_MAX(quadtree->maxResultsPerQueryInUse, unsigned int, queryResults.numResults);
	#endif
			}

			else
			{
				unsigned int baseIndex = (index * 4);
				unsigned int newOffset = offset + levelWidth;
				unsigned int newLevelWidth = levelWidth * 4;
				
				stack.push(baseIndex, newOffset, newLevelWidth);
				stack.push(baseIndex + 1, newOffset, newLevelWidth);
				stack.push(baseIndex + 2, newOffset, newLevelWidth);
				stack.push(baseIndex + 3, newOffset, newLevelWidth);
			}
		}
#ifdef COLLECT_STATISTICS
		ATOMIC_ADD(quadtree->numCollisionChecks, unsigned int, 1);
#endif
	}
}

#ifdef COLLECT_STATISTICS
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