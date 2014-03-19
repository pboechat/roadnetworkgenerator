#ifndef COLLISION_DETECTION_KERNEL_CUH
#define COLLISION_DETECTION_KERNEL_CUH

#include <Constants.h>
#include <CpuGpuCompatibility.h>
#include <Procedures.h>
#include <Graph.h>
#include <QuadrantEdges.h>

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void computeCollisionsInQuadrant(Graph* graph, QuadrantEdges* quadrantEdges)
{
	unsigned int i = THREAD_IDX_X;
	while (i < quadrantEdges->lastEdgeIndex)
	{
		Edge& thisEdge = graph->edges[quadrantEdges->edges[i]];

#ifdef PARALLEL
		while(!thisEdge.readFlag);
#endif

		int j = i - 1;
		while (j >= 0)
		{
			Edge& otherEdge = graph->edges[quadrantEdges->edges[j]];

#ifdef PARALLEL
			while(!otherEdge.readFlag);
#endif

			bool tryAgain;
			do
			{
				tryAgain = false;
				vml_vec2 intersection;
				if (checkIntersection(graph, thisEdge, otherEdge, intersection))
				{
					tryAgain = true;
					if (ATOMIC_EXCH(thisEdge.owner, int, THREAD_IDX_X) == -1)
					{
						if (ATOMIC_EXCH(otherEdge.owner, int, THREAD_IDX_X) == -1)
						{
							int newVertexIndex = createVertex(graph, intersection);
							splitEdge(graph, otherEdge, newVertexIndex);
							splitEdge(graph, thisEdge, newVertexIndex);
							THREADFENCE();
							tryAgain = false;
							ATOMIC_EXCH(otherEdge.owner, int, -1);
						}

						ATOMIC_EXCH(thisEdge.owner, int, -1);	
					}
				}
#ifdef COLLECT_STATISTICS
				ATOMIC_ADD(graph->numCollisionChecks, unsigned int, 1);
#endif
			} while (tryAgain);

			j--;
		}

		i += BLOCK_DIM_X;
	}
}

#ifdef PARALLEL
//////////////////////////////////////////////////////////////////////////
__global__ void collisionDetectionKernel(Graph* graph)
{
	__shared__ QuadrantEdges* quadrantEdges;

	if (threadIdx.x == 0)
	{
		quadrantEdges = &graph->quadtree->quadrantsEdges[blockIdx.x % graph->quadtree->numLeafQuadrants];
	}

	__syncthreads();

	computeCollisionsInQuadrant(graph, quadrantEdges);
}
#else
//////////////////////////////////////////////////////////////////////////
void collisionDetectionKernel(Graph* graph)
{
	QuadTree* quadtree = graph->quadtree;
	for (unsigned int i = 0; i < quadtree->numLeafQuadrants; i++)
	{
		computeCollisionsInQuadrant(graph, &quadtree->quadrantsEdges[i]);
	}
}
#endif

#endif