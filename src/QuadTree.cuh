#ifndef ROADNETWORKGRAPH_QUADREE_CUH
#define ROADNETWORKGRAPH_QUADREE_CUH

#include "Defines.h"
#include <Quadrant.cuh>
#include <QuadrantEdges.cuh>

#include <Line2D.cuh>
#include <Circle2D.cuh>
#include <Box2D.cuh>

#include <vector_math.h>

namespace RoadNetworkGraph
{

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
	QuadrantEdgesIndex numQuadrantEdges;
#ifdef _DEBUG
	unsigned long numCollisionChecks;
	unsigned int maxEdgesPerQuadrantInUse;
	unsigned int maxResultsPerQueryInUse;
#endif

};

/*
//////////////////////////////////////////////////////////////////////////
GLOBAL_CODE void initializeQuadtree(QuadTree* quadtree, const Box2D& worldBounds, unsigned int depth, unsigned int maxResultsPerQuery, unsigned int maxQuadrants, Quadrant* quadrants, QuadrantEdges* quadrantEdges);
//////////////////////////////////////////////////////////////////////////
GLOBAL_CODE void updateNonPointerFields(QuadTree* quadtree, QuadrantEdgesIndex numQuadrantEdges, unsigned int maxResultsPerQuery, Box2D worldBounds, unsigned int maxDepth, unsigned int maxQuadrants, unsigned int totalNumQuadrants, unsigned int numLeafQuadrants
#ifdef _DEBUG
	, unsigned long numCollisionChecks
	, unsigned int maxEdgesPerQuadrantInUse
	, unsigned int maxResultsPerQueryInUse
#endif	
);
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
*/

//////////////////////////////////////////////////////////////////////////
HOST_AND_DEVICE_CODE void initializeQuadrant(QuadTree* quadtree, const Box2D& quadrantBounds, unsigned int depth = 0, unsigned int index = 0, unsigned int offset = 0, unsigned int levelWidth = 1);
//////////////////////////////////////////////////////////////////////////
HOST_AND_DEVICE_CODE void removeEdgeReferencesInVertices(QuadrantEdges* quadrantEdges, EdgeIndex edgeIndex);

//////////////////////////////////////////////////////////////////////////
GLOBAL_CODE void initializeQuadtreeOnDevice(QuadTree* quadtree, Box2D worldBounds, unsigned int depth, unsigned int maxResultsPerQuery, unsigned int maxQuadrants, Quadrant* quadrants, QuadrantEdges* quadrantEdges)
{
	quadtree->worldBounds = worldBounds;
	quadtree->maxDepth = depth;
	quadtree->maxResultsPerQuery = maxResultsPerQuery;
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
HOST_CODE void initializeQuadtreeOnHost(QuadTree* quadtree, Box2D worldBounds, unsigned int depth, unsigned int maxResultsPerQuery, unsigned int maxQuadrants, Quadrant* quadrants, QuadrantEdges* quadrantEdges)
{
	quadtree->worldBounds = worldBounds;
	quadtree->maxDepth = depth;
	quadtree->maxResultsPerQuery = maxResultsPerQuery;
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
GLOBAL_CODE void updateNonPointerFields(QuadTree* quadtree, QuadrantEdgesIndex numQuadrantEdges, unsigned int maxResultsPerQuery, Box2D worldBounds, unsigned int maxDepth, unsigned int maxQuadrants, unsigned int totalNumQuadrants, unsigned int numLeafQuadrants
#ifdef _DEBUG
	, unsigned long numCollisionChecks
	, unsigned int maxEdgesPerQuadrantInUse
	, unsigned int maxResultsPerQueryInUse
#endif
)
{
	quadtree->numQuadrantEdges = numQuadrantEdges;
	quadtree->maxResultsPerQuery = maxResultsPerQuery;
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

	Quadrant& quadrant = quadtree->quadrants[i];
	quadrant.depth = depth;
	quadrant.bounds = quadrantBounds;

	if (depth == quadtree->maxDepth - 1) // leaf
	{
		quadrant.edges = quadtree->numQuadrantEdges++;
		return;
	}

	unsigned int baseIndex = (index * 4);
	unsigned int newOffset = offset + levelWidth;
	unsigned int newLevelWidth = levelWidth * 4;
	unsigned int newDepth = depth + 1;
	vml_vec2 subQuadrantSize = quadrantBounds.getExtents() / 2.0f;

	for (unsigned int y = 0, i = 0; y < 2; y++)
	{
		float subQuadrantY = quadrantBounds.getMin().y + ((float)y * subQuadrantSize.y);

		for (unsigned int x = 0; x < 2; x++, i++)
		{
			initializeQuadrant(quadtree, Box2D(quadrantBounds.getMin().x + ((float)x * subQuadrantSize.x), subQuadrantY, subQuadrantSize.x, subQuadrantSize.y), newDepth, baseIndex + i, newOffset, newLevelWidth);
		}
	}
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void insert(QuadTree* quadtree, EdgeIndex edgeIndex, const Line2D& edgeLine, unsigned int index = 0, unsigned int offset = 0, unsigned int levelWidth = 1)
{
	Quadrant& quadrant = quadtree->quadrants[offset + index];

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

			// FIXME: checking boundaries
			if (quadrantEdges.lastEdgeIndex >= MAX_EDGES_PER_QUADRANT)
			{
				THROW_EXCEPTION("max. edges per quadrant overflow");
			}

			quadrantEdges.edges[quadrantEdges.lastEdgeIndex++] = edgeIndex;

#ifdef _DEBUG
			quadtree->maxEdgesPerQuadrantInUse = MathExtras::max<unsigned int>(quadtree->maxEdgesPerQuadrantInUse, quadrantEdges.lastEdgeIndex);
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
	quadtree->numCollisionChecks++;
#endif
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void remove(QuadTree* quadtree, EdgeIndex edgeIndex, const Line2D& edgeLine, unsigned int index = 0, unsigned int offset = 0, unsigned int levelWidth = 1)
{
	Quadrant& quadrant = quadtree->quadrants[offset + index];

	if (quadrant.bounds.isIntersected(edgeLine))
	{
		if (quadrant.depth == quadtree->maxDepth - 1)
		{
			// FIXME: checking invariants
			if (quadrant.edges == -1)
			{
				THROW_EXCEPTION("quadrant.edges == -1");
			}

			removeEdgeReferencesInVertices(&quadtree->quadrantsEdges[quadrant.edges], edgeIndex);
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
	quadtree->numCollisionChecks++;
#endif
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void removeEdgeReferencesInVertices(QuadrantEdges* quadrantEdges, EdgeIndex edgeIndex)
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
DEVICE_CODE void query(QuadTree* quadtree, const T& shape, EdgeIndex* queryResult, unsigned int& size, unsigned int offset = 0)
{
	size = offset;
	recursiveQuery(quadtree, shape, queryResult, size, 0, 0, 1);
}
//////////////////////////////////////////////////////////////////////////
template<typename T>
DEVICE_CODE void recursiveQuery(QuadTree* quadtree, const T& shape, EdgeIndex* queryResult, unsigned int& size, unsigned int index, unsigned int offset, unsigned int levelWidth)
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
			quadtree->maxResultsPerQueryInUse = MathExtras::max<unsigned int>(quadtree->maxResultsPerQueryInUse, size);
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