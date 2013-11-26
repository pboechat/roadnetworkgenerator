#ifndef QUADREE_H
#define QUADREE_H

#include <Defines.h>
#include <Quadrant.h>
#include <QuadrantEdges.h>
#include <Line.h>
#include <Circle.h>
#include <AABB.h>
#include <glm/glm.hpp>

#include <cmath>
#include <exception>

#ifndef max
#define max(a, b) (((a) > (b)) ? (a) : (b))
#endif

namespace RoadNetworkGraph
{

class QuadTree
{
public:
	QuadTree(const AABB& worldBounds, unsigned int maxDepth) : worldBounds(worldBounds), maxDepth(maxDepth), lastQuadrantIndex(0), quadrants(0), quadrantsEdges(0), totalNumQuadrants(0)
	{
		unsigned int numLeafQuadrants;
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

		glm::vec3 quadrantSize = worldBounds.getExtents();
		QuadrantIndex quadrantIndex = 0;
		QuadrantEdgesIndex quadrantEdgesIndex = 0;
		for (unsigned int depth = 0, side = 1; depth < maxDepth; depth++, side *= 2)
		{
			bool leaf = (depth == maxDepth - 1);
			for (unsigned int y = 0; y < side; y++)
			{
				float boundsY = worldBounds.min.y + ((float)y * quadrantSize.y);
				for (unsigned int x = 0; x < side; x++, quadrantIndex++)
				{
					Quadrant& quadrant = quadrants[quadrantIndex];
					quadrant.depth = depth;
					quadrant.bounds = AABB(worldBounds.min.x + ((float)x * quadrantSize.x), boundsY, quadrantSize.x, quadrantSize.y);

					if (leaf)
					{
						// assign edge storage reference (quadrant edges) only to leaf quadrants
						quadrant.edges = quadrantEdgesIndex++;
					}
				}
			}
			quadrantSize = quadrantSize / 2.0f;
		}
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

	void query(const AABB& region, EdgeIndex* queryResult, unsigned int& size, unsigned int offset = 0) const
	{
		// TODO: optimize
		size = 0;
		for (unsigned int i = 0; i < totalNumQuadrants; i++)
		{
			Quadrant& quadrant = quadrants[i];

			if (quadrant.depth != maxDepth - 1)
			{
				continue;
			}

			if (region.intersects(quadrant.bounds))
			{
				QuadrantEdges& quadrantEdges = quadrantsEdges[quadrant.edges];
				
				for (unsigned int i = 0; i < quadrantEdges.lastEdgeIndex; i++)
				{
					queryResult[size++] = quadrantEdges.edges[i];

					// FIXME: checking boundaries
					if (size >= MAX_RESULTS_PER_QUERY)
					{
						throw std::exception("max. results per query overflow");
					}
				}
			}
		}
	}

	void query(const Circle& circle, EdgeIndex* queryResult, unsigned int& size, unsigned int offset = 0) const
	{
		// TODO: optimize
		size = offset;
		for (unsigned int i = 0; i < totalNumQuadrants; i++)
		{
			Quadrant& quadrant = quadrants[i];

			if (quadrant.depth != maxDepth - 1)
			{
				continue;
			}

			if (quadrant.bounds.intersects(circle))
			{
				QuadrantEdges& quadrantEdges = quadrantsEdges[quadrant.edges];

				for (unsigned int i = 0; i < quadrantEdges.lastEdgeIndex; i++)
				{
					queryResult[size++] = quadrantEdges.edges[i];

					// FIXME: checking boundaries
					if (size >= MAX_RESULTS_PER_QUERY)
					{
						throw std::exception("max. results per query overflow");
					}
				}
			}
		}
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

	void remove(EdgeIndex edgeIndex, const Line& edgeLine)
	{
		// TODO: optimize
		for (unsigned int i = 0; i < totalNumQuadrants; i++)
		{
			Quadrant& quadrant = quadrants[i];

			if (quadrant.depth != maxDepth - 1)
			{
				continue;
			}

			if (quadrant.bounds.isIntersected(edgeLine))
			{
				removeEdge(&quadrantsEdges[quadrant.edges], edgeIndex);
			}
		}
	}

	void insert(EdgeIndex edgeIndex, const Line& edgeLine)
	{
		// TODO: optimize
		for (unsigned int i = 0; i < totalNumQuadrants; i++)
		{
			Quadrant& quadrant = quadrants[i];

			if (quadrant.depth != maxDepth - 1)
			{
				continue;
			}

			if (quadrant.bounds.isIntersected(edgeLine))
			{
				QuadrantEdges& quadrantEdges = quadrantsEdges[quadrant.edges];

				// FIXME: checking boundaries
				if (quadrantEdges.lastEdgeIndex == MAX_EDGES_PER_QUADRANT)
				{
					throw std::exception("max. edges per quadrant overflow");
				}

				quadrantEdges.edges[quadrantEdges.lastEdgeIndex++] = edgeIndex;
			}
		}

		/*unsigned int collisionMask = 0xffffffff;
		QuadrantIndex index = 0;
		for (unsigned int depth = 0, numQuadrantsDepth = 1; depth < maxDepth; depth++, numQuadrantsDepth *= 4)
		{
			unsigned int newCollisionMask = 0;

			unsigned int shift = max(1, 32 / numQuadrantsDepth);
			unsigned int maskSide = max(1, numQuadrantsDepth / 32);
			unsigned int maskArea = maskSide * maskSide;
			unsigned int baseMask = ((unsigned long long)1 << shift) - 1;

			unsigned int depthIndex = 0;
			unsigned int maskIndex = 0;
			while (depthIndex < numQuadrantsDepth)
			{
				unsigned int quadrantMask = baseMask << (shift * maskIndex++);
				if ((collisionMask & quadrantMask) != 0)
				{
					for (unsigned int y = 0; y < maskSide; y++)
					{
						for (unsigned int x = 0; x < maskSide; x++)
						{
							Quadrant& quadrant = quadrants[index++];
							if (quadrant.bounds.isIntersected(edgeLine))
							{
								quadrant.edges[quadrant.lastEdgeIndex++] = edgeIndex;
								newCollisionMask |= quadrantMask;
							}
						}
					}
				}
				else
				{
					index += maskArea;
				}
				depthIndex += maskArea;
			}

			collisionMask = newCollisionMask;
		}*/
	}

private:
	AABB worldBounds;
	unsigned int maxDepth;
	Quadrant* quadrants;
	QuadrantEdges* quadrantsEdges;
	QuadrantIndex lastQuadrantIndex;
	unsigned int totalNumQuadrants;

};

}

#endif