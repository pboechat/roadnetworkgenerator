#ifndef ROADNETWORKGRAPH_QUADREE_H
#define ROADNETWORKGRAPH_QUADREE_H

#include <Defines.h>
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

struct QuadTree
{
	unsigned int maxResultsPerQuery;
	Box2D worldBounds;
	unsigned int maxDepth;
	Quadrant* quadrants;
	QuadrantEdges* quadrantsEdges;
	unsigned int totalNumQuadrants;
	unsigned int numLeafQuadrants;
	QuadrantEdgesIndex numQuadrantEdges;
#ifdef _DEBUG
	unsigned long numCollisionChecks;
#endif

};

//////////////////////////////////////////////////////////////////////////
void initializeQuadtree(QuadTree* quadtree, const Box2D& worldBounds, unsigned int depth, unsigned int maxResultsPerQuery, Quadrant* quadrants, QuadrantEdges* quadrantEdges);
//////////////////////////////////////////////////////////////////////////
void insert(QuadTree* quadtree, EdgeIndex edgeIndex, const Line2D& edgeLine, unsigned int index = 0, unsigned int offset = 0, unsigned int levelWidth = 1);
//////////////////////////////////////////////////////////////////////////
void remove(QuadTree* quadtree, EdgeIndex edgeIndex, const Line2D& edgeLine, unsigned int index = 0, unsigned int offset = 0, unsigned int levelWidth = 1);
//////////////////////////////////////////////////////////////////////////
template<typename T>
void query(QuadTree* quadtree, const T& shape, EdgeIndex* queryResult, unsigned int& size, unsigned int offset = 0)
{
	size = offset;
	recurssiveQuery(quadtree, shape, queryResult, size, 0, 0, 1);
}
//////////////////////////////////////////////////////////////////////////
template<typename T>
void recurssiveQuery(QuadTree* quadtree, const T& shape, EdgeIndex* queryResult, unsigned int& size, unsigned int index, unsigned int offset, unsigned int levelWidth)
{
	Quadrant& quadrant = quadtree->quadrants[offset + index];

	if (quadrant.bounds.intersects(shape))
	{
		if (quadrant.depth == quadtree->maxDepth - 1)
		{
			// FIXME: checking invariants
			if (quadrant.edges == -1)
			{
				throw std::exception("quadrant.edges == -1");
			}

			QuadrantEdges& quadrantEdges = quadtree->quadrantsEdges[quadrant.edges];

			for (unsigned int i = 0; i < quadrantEdges.lastEdgeIndex; i++)
			{
				queryResult[size++] = quadrantEdges.edges[i];

				// FIXME: checking boundaries
				if (size >= quadtree->maxResultsPerQuery)
				{
					throw std::exception("max. results per query overflow");
				}
			}
		}

		else
		{
			unsigned int baseIndex = (index * 4);
			unsigned int newOffset = offset + levelWidth;
			unsigned int newLevelWidth = levelWidth * 4;
			recurssiveQuery(quadtree, shape, queryResult, size, baseIndex, newOffset, newLevelWidth);
			recurssiveQuery(quadtree, shape, queryResult, size, baseIndex + 1, newOffset, newLevelWidth);
			recurssiveQuery(quadtree, shape, queryResult, size, baseIndex + 2, newOffset, newLevelWidth);
			recurssiveQuery(quadtree, shape, queryResult, size, baseIndex + 3, newOffset, newLevelWidth);
		}
	}

#ifdef _DEBUG
	quadtree->numCollisionChecks++;
#endif
}

#ifdef _DEBUG
//////////////////////////////////////////////////////////////////////////
unsigned int getAllocatedMemory(QuadTree* quadtree);
//////////////////////////////////////////////////////////////////////////
unsigned int getMemoryInUse(QuadTree* quadtree);
//////////////////////////////////////////////////////////////////////////
unsigned long getNumCollisionChecks(QuadTree* quadtree);
//////////////////////////////////////////////////////////////////////////
unsigned int getMaxEdgesPerQuadrantInUse(QuadTree* quadtree);
#endif

}

#endif