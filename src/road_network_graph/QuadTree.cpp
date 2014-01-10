#include <QuadTree.h>

namespace RoadNetworkGraph
{

//////////////////////////////////////////////////////////////////////////
void initializeQuadrant(QuadTree* quadtree, const Box2D& quadrantBounds, unsigned int depth = 0, unsigned int index = 0, unsigned int offset = 0, unsigned int levelWidth = 1);
//////////////////////////////////////////////////////////////////////////
void removeEdgeReferencesInVertices(QuadrantEdges* quadrantEdges, EdgeIndex edgeIndex);

//////////////////////////////////////////////////////////////////////////
void initializeQuadtree(QuadTree* quadtree, const Box2D& worldBounds, unsigned int depth, unsigned int maxResultsPerQuery, Quadrant* quadrants, QuadrantEdges* quadrantEdges)
{
	quadtree->worldBounds = worldBounds;
	quadtree->maxDepth = depth;
	quadtree->maxResultsPerQuery = maxResultsPerQuery;
	quadtree->quadrants = quadrants;
	quadtree->quadrantsEdges = quadrantEdges;
	quadtree->numQuadrantEdges = 0;
	quadtree->numCollisionChecks = 0;
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
void initializeQuadrant(QuadTree* quadtree, const Box2D& quadrantBounds, unsigned int depth, unsigned int index, unsigned int offset, unsigned int levelWidth)
{
	Quadrant& quadrant = quadtree->quadrants[offset + index];
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
		float subQuadrantY = quadrantBounds.min.y + ((float)y * subQuadrantSize.y);

		for (unsigned int x = 0; x < 2; x++, i++)
		{
			initializeQuadrant(quadtree, Box2D(quadrantBounds.min.x + ((float)x * subQuadrantSize.x), subQuadrantY, subQuadrantSize.x, subQuadrantSize.y), newDepth, baseIndex + i, newOffset, newLevelWidth);
		}
	}
}

//////////////////////////////////////////////////////////////////////////
void insert(QuadTree* quadtree, EdgeIndex edgeIndex, const Line2D& edgeLine, unsigned int index, unsigned int offset, unsigned int levelWidth)
{
	Quadrant& quadrant = quadtree->quadrants[offset + index];

	if (quadrant.bounds.isIntersected(edgeLine))
	{
		if (quadrant.depth == quadtree->maxDepth - 1)
		{
			// FIXME: checking invariants
			if (quadrant.edges == -1)
			{
				throw std::exception("quadrant.edges == -1");
			}

			QuadrantEdges& quadrantEdges = quadtree->quadrantsEdges[quadrant.edges];

			// FIXME: checking boundaries
			if (quadrantEdges.lastEdgeIndex == MAX_EDGES_PER_QUADRANT)
			{
				throw std::exception("max. edges per quadrant overflow");
			}

			quadrantEdges.edges[quadrantEdges.lastEdgeIndex++] = edgeIndex;
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
void remove(QuadTree* quadtree, EdgeIndex edgeIndex, const Line2D& edgeLine, unsigned int index, unsigned int offset, unsigned int levelWidth)
{
	Quadrant& quadrant = quadtree->quadrants[offset + index];

	if (quadrant.bounds.isIntersected(edgeLine))
	{
		if (quadrant.depth == quadtree->maxDepth - 1)
		{
			// FIXME: checking invariants
			if (quadrant.edges == -1)
			{
				throw std::exception("quadrant.edges == -1");
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
void removeEdgeReferencesInVertices(QuadrantEdges* quadrantEdges, EdgeIndex edgeIndex)
{
	// FIXME: checking invariants
	if (quadrantEdges == 0)
	{
		throw std::exception("quadrantEdges == 0");
	}

	// FIXME: checking boundaries
	if (quadrantEdges->lastEdgeIndex == 0)
	{
		throw std::exception("tried to remove edge from an empty quadrant");
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
		throw std::exception("!found");
	}

	for (unsigned int j = i; j < quadrantEdges->lastEdgeIndex - 1; j++)
	{
		quadrantEdges->edges[j] = quadrantEdges->edges[j + 1];
	}

	quadrantEdges->lastEdgeIndex--;
}

#ifdef _DEBUG
//////////////////////////////////////////////////////////////////////////
unsigned int getAllocatedMemory(QuadTree* quadtree)
{
	unsigned int quadrantsBufferMemory = quadtree->totalNumQuadrants * sizeof(Quadrant);
	unsigned int quadrantsEdgesBufferMemory = quadtree->numLeafQuadrants * sizeof(QuadrantEdges);
	return (quadrantsBufferMemory + quadrantsEdgesBufferMemory);
}

//////////////////////////////////////////////////////////////////////////
unsigned int getMemoryInUse(QuadTree* quadtree)
{
	unsigned int quadrantsBufferMemoryInUse = quadtree->totalNumQuadrants * sizeof(Quadrant);
	unsigned int quadrantsEdgesBufferMemoryInUse = 0;

	for (unsigned int i = 0; i < quadtree->numLeafQuadrants; i++)
	{
		QuadrantEdges& quadrantEdges = quadtree->quadrantsEdges[i];
		quadrantsEdgesBufferMemoryInUse += sizeof(EdgeIndex) * quadrantEdges.lastEdgeIndex + sizeof(unsigned int);
	}

	return (quadrantsBufferMemoryInUse + quadrantsEdgesBufferMemoryInUse);
}

//////////////////////////////////////////////////////////////////////////
unsigned long getNumCollisionChecks(QuadTree* quadtree)
{
	return quadtree->numCollisionChecks;
}

//////////////////////////////////////////////////////////////////////////
unsigned int getMaxEdgesPerQuadrantInUse(QuadTree* quadtree)
{
	unsigned int maxEdgesPerQuadrantInUse = 0;

	for (unsigned int i = 0; i < quadtree->numLeafQuadrants; i++)
	{
		if (quadtree->quadrantsEdges[i].lastEdgeIndex > maxEdgesPerQuadrantInUse)
		{
			maxEdgesPerQuadrantInUse = quadtree->quadrantsEdges[i].lastEdgeIndex;
		}
	}

	return maxEdgesPerQuadrantInUse;
}
#endif

}