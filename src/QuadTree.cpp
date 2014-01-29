#include <QuadTree.cuh>

namespace RoadNetworkGraph
{
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
		quadrantsEdgesBufferMemoryInUse += sizeof(EdgeIndex) * quadrantEdges.lastEdgeIndex + sizeof(unsigned int);
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

}