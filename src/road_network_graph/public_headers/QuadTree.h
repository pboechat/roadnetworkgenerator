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

class QuadTree
{
public:
	QuadTree(const Box2D& worldBounds, unsigned int maxDepth, unsigned int maxResultsPerQuery) :
		worldBounds(worldBounds),
		maxDepth(maxDepth),
		quadrants(0),
		quadrantsEdges(0),
		totalNumQuadrants(0),
		numLeafQuadrants(0),
		maxResultsPerQuery(maxResultsPerQuery),
		lastQuadrantEdgesIndex(0)
#ifdef _DEBUG
		, numCollisionChecks(0)
#endif
	{
		totalNumQuadrants = 0;

		for (unsigned int i = 0; i < maxDepth; i++)
		{
			unsigned int numQuadrants = (unsigned int)pow(4.0f, (int)i);

			if (i == maxDepth - 1)
			{
				numLeafQuadrants = numQuadrants;
			}

			totalNumQuadrants += numQuadrants;
		}

		quadrants = new Quadrant[totalNumQuadrants];
		quadrantsEdges = new QuadrantEdges[numLeafQuadrants];
		initializeQuadrant(worldBounds);
	}

	~QuadTree()
	{
		if (quadrants != 0)
		{
			delete[] quadrants;
		}

		if (quadrantsEdges != 0)
		{
			delete[] quadrantsEdges;
		}
	}

	template<typename T>
	void query(const T& shape, EdgeIndex* queryResult, unsigned int& size, unsigned int offset = 0)
	{
		size = offset;
		query(shape, queryResult, size, 0, 0, 1);
	}

	void remove(EdgeIndex edgeIndex, const Line2D& edgeLine, unsigned int index = 0, unsigned int offset = 0, unsigned int levelWidth = 1)
	{
		Quadrant& quadrant = quadrants[offset + index];

		if (quadrant.bounds.isIntersected(edgeLine))
		{
			if (quadrant.depth == maxDepth - 1)
			{
				// FIXME: checking invariants
				if (quadrant.edges == -1)
				{
					throw std::exception("quadrant.edges == -1");
				}

				removeEdge(&quadrantsEdges[quadrant.edges], edgeIndex);
			}

			else
			{
				unsigned int baseIndex = (index * 4);
				unsigned int newOffset = offset + levelWidth;
				unsigned int newLevelWidth = levelWidth * 4;
				insert(edgeIndex, edgeLine, baseIndex, newOffset, newLevelWidth);
				insert(edgeIndex, edgeLine, baseIndex + 1, newOffset, newLevelWidth);
				insert(edgeIndex, edgeLine, baseIndex + 2, newOffset, newLevelWidth);
				insert(edgeIndex, edgeLine, baseIndex + 3, newOffset, newLevelWidth);
			}
		}

#ifdef _DEBUG
		numCollisionChecks++;
#endif
	}

	void insert(EdgeIndex edgeIndex, const Line2D& edgeLine, unsigned int index = 0, unsigned int offset = 0, unsigned int levelWidth = 1)
	{
		Quadrant& quadrant = quadrants[offset + index];

		if (quadrant.bounds.isIntersected(edgeLine))
		{
			if (quadrant.depth == maxDepth - 1)
			{
				// FIXME: checking invariants
				if (quadrant.edges == -1)
				{
					throw std::exception("quadrant.edges == -1");
				}

				QuadrantEdges& quadrantEdges = quadrantsEdges[quadrant.edges];

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
				insert(edgeIndex, edgeLine, baseIndex, newOffset, newLevelWidth);
				insert(edgeIndex, edgeLine, baseIndex + 1, newOffset, newLevelWidth);
				insert(edgeIndex, edgeLine, baseIndex + 2, newOffset, newLevelWidth);
				insert(edgeIndex, edgeLine, baseIndex + 3, newOffset, newLevelWidth);
			}
		}

#ifdef _DEBUG
		numCollisionChecks++;
#endif
	}

#ifdef _DEBUG
	unsigned int getAllocatedMemory() const
	{
		unsigned int quadrantsBufferMemory = totalNumQuadrants * sizeof(Quadrant);
		unsigned int quadrantsEdgesBufferMemory = numLeafQuadrants * sizeof(QuadrantEdges);
		return (quadrantsBufferMemory + quadrantsEdgesBufferMemory);
	}

	unsigned int getMemoryInUse() const
	{
		unsigned int quadrantsBufferMemoryInUse = totalNumQuadrants * sizeof(Quadrant);
		unsigned int quadrantsEdgesBufferMemoryInUse = 0;

		for (unsigned int i = 0; i < numLeafQuadrants; i++)
		{
			QuadrantEdges& quadrantEdges = quadrantsEdges[i];
			quadrantsEdgesBufferMemoryInUse += sizeof(EdgeIndex) * quadrantEdges.lastEdgeIndex + sizeof(unsigned int);
		}

		return (quadrantsBufferMemoryInUse + quadrantsEdgesBufferMemoryInUse);
	}

	unsigned long getNumCollisionChecks() const
	{
		return numCollisionChecks;
	}

	unsigned int getMaxEdgesPerQuadrantInUse() const
	{
		unsigned int maxEdgesPerQuadrantInUse = 0;

		for (unsigned int i = 0; i < numLeafQuadrants; i++)
		{
			if (quadrantsEdges[i].lastEdgeIndex > maxEdgesPerQuadrantInUse)
			{
				maxEdgesPerQuadrantInUse = quadrantsEdges[i].lastEdgeIndex;
			}
		}

		return maxEdgesPerQuadrantInUse;
	}
#endif

private:
	unsigned int maxResultsPerQuery;
	Box2D worldBounds;
	unsigned int maxDepth;
	Quadrant* quadrants;
	QuadrantEdges* quadrantsEdges;
	unsigned int totalNumQuadrants;
	unsigned int numLeafQuadrants;
	QuadrantEdgesIndex lastQuadrantEdgesIndex;
#ifdef _DEBUG
	unsigned long numCollisionChecks;
#endif

	void initializeQuadrant(const Box2D& quadrantBounds, unsigned int depth = 0, unsigned int index = 0, unsigned int offset = 0, unsigned int levelWidth = 1)
	{
		Quadrant& quadrant = quadrants[offset + index];
		quadrant.depth = depth;
		quadrant.bounds = quadrantBounds;

		if (depth == maxDepth - 1) // leaf
		{
			quadrant.edges = lastQuadrantEdgesIndex++;
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
				initializeQuadrant(Box2D(quadrantBounds.min.x + ((float)x * subQuadrantSize.x), subQuadrantY, subQuadrantSize.x, subQuadrantSize.y), newDepth, baseIndex + i, newOffset, newLevelWidth);
			}
		}
	}

	template<typename T>
	void query(const T& shape, EdgeIndex* queryResult, unsigned int& size, unsigned int index, unsigned int offset, unsigned int levelWidth)
	{
		Quadrant& quadrant = quadrants[offset + index];

		if (quadrant.bounds.intersects(shape))
		{
			if (quadrant.depth == maxDepth - 1)
			{
				// FIXME: checking invariants
				if (quadrant.edges == -1)
				{
					throw std::exception("quadrant.edges == -1");
				}

				QuadrantEdges& quadrantEdges = quadrantsEdges[quadrant.edges];

				for (unsigned int i = 0; i < quadrantEdges.lastEdgeIndex; i++)
				{
					queryResult[size++] = quadrantEdges.edges[i];

					// FIXME: checking boundaries
					if (size >= maxResultsPerQuery)
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
				query(shape, queryResult, size, baseIndex, newOffset, newLevelWidth);
				query(shape, queryResult, size, baseIndex + 1, newOffset, newLevelWidth);
				query(shape, queryResult, size, baseIndex + 2, newOffset, newLevelWidth);
				query(shape, queryResult, size, baseIndex + 3, newOffset, newLevelWidth);
			}
		}

#ifdef _DEBUG
		numCollisionChecks++;
#endif
	}

	void removeEdge(QuadrantEdges* quadrantEdges, EdgeIndex edgeIndex)
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

};

}

#endif