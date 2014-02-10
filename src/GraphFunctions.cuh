#ifndef GRAPHFUNCTIONS_CUH
#define GRAPHFUNCTIONS_CUH

#pragma once

#include <Constants.h>
#include <CpuGpuCompatibility.h>
#include <Graph.h>
#include <GraphTraversal.h>
#include <Box2D.h>
#include <Line2D.h>
#include <Circle2D.h>
#include <VectorMath.h>
#include <QuadTree.h>
#include <QueryResults.h>
#include <QuadTreeFunctions.cuh>
#include <VertexFunctions.cuh>
#include <Primitive.h>
#ifdef USE_CUDA
#include <cutil.h>
#endif

//////////////////////////////////////////////////////////////////////////
enum IntersectionType
{
	NONE,
	SOURCE_SOURCE_SNAPPING,
	SOURCE_DESTINATION_SNAPPING,
	DESTINATION_SOURCE_SNAPPING,
	DESTINATION_DESTINATION_SNAPPING,
	EDGE_INTERSECTION
};

//////////////////////////////////////////////////////////////////////////
GLOBAL_CODE void initializeGraphOnDevice(Graph* graph, float snapRadius, unsigned int maxVertices, unsigned int maxEdges, Vertex* vertices, Edge* edges, QuadTree* quadtree)
{
	graph->numVertices = 0;
	graph->numEdges = 0;
	graph->vertices = vertices;
	graph->edges = edges;
	graph->maxVertices = maxVertices;
	graph->maxEdges = maxEdges;
	graph->snapRadius = snapRadius;
	graph->quadtree = quadtree;
#ifdef COLLECT_STATISTICS
	graph->numCollisionChecks = 0;
#endif
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE void initializeGraphOnHost(Graph* graph, float snapRadius, unsigned int maxVertices, unsigned int maxEdges, Vertex* vertices, Edge* edges, QuadTree* quadtree)
{
	graph->numVertices = 0;
	graph->numEdges = 0;
	graph->vertices = vertices;
	graph->edges = edges;
	graph->maxVertices = maxVertices;
	graph->maxEdges = maxEdges;
	graph->snapRadius = snapRadius;
	graph->quadtree = quadtree;
#ifdef COLLECT_STATISTICS
	graph->numCollisionChecks = 0;
#endif
}

//////////////////////////////////////////////////////////////////////////
GLOBAL_CODE void updateNonPointerFields(Graph* graph, unsigned int numVertices, unsigned int numEdges
#ifdef COLLECT_STATISTICS
	, unsigned long numCollisionChecks
#endif	
)
{
	graph->numVertices = numVertices;
	graph->numEdges = numEdges;
#ifdef COLLECT_STATISTICS
	graph->numCollisionChecks = numCollisionChecks;
#endif
}

#ifdef USE_CUDA
//////////////////////////////////////////////////////////////////////////
__device__ __host__ int createVertex(Graph* graph, const vml_vec2& position)
{
	int newVertexIndex;
#ifdef __CUDA_ARCH__ 
	unsigned int mask = __ballot(1);
	unsigned int numberOfActiveThreads = __popc(mask);
	int laneId = __popc(lanemask_lt() & mask);
	int leadingThreadId = __ffs(mask) - 1;

	int oldValue;
	if (laneId == 0)
	{
		oldValue = atomicAdd((int*)&graph->numVertices, numberOfActiveThreads);

		// FIXME: checking boundaries
		if (graph->numVertices >= (int)graph->maxVertices)
		{
			THROW_EXCEPTION("max. vertices overflow");
		}
	}

	oldValue = __shfl(oldValue, leadingThreadId);

	newVertexIndex = oldValue + laneId;
#else
	newVertexIndex = ATOMIC_ADD(graph->numVertices, int, 1);

	// FIXME: checking boundaries
	if (graph->numVertices >= (int)graph->maxVertices)
	{
		THROW_EXCEPTION("max. vertices overflow");
	}
#endif

	Vertex& newVertex = graph->vertices[newVertexIndex];

	newVertex.index = newVertexIndex;
	newVertex.setPosition(position);

	return newVertexIndex;
}
#else
//////////////////////////////////////////////////////////////////////////
int createVertex(Graph* graph, const vml_vec2& position)
{
	// FIXME: checking boundaries
	if (graph->numVertices >= (int)graph->maxVertices)
	{
		THROW_EXCEPTION("max. vertices overflow");
	}

	int newVertexIndex = ATOMIC_ADD(graph->numVertices, int, 1);

	Vertex& newVertex = graph->vertices[newVertexIndex];

	newVertex.index = newVertexIndex;
	newVertex.setPosition(position);

	return newVertexIndex;
}
#endif

//////////////////////////////////////////////////////////////////////////
HOST_AND_DEVICE_CODE int connect(Graph* graph, int sourceVertexIndex, int destinationVertexIndex, char attr1 = 0, char attr2 = 0, char attr3 = 0, char attr4 = 0)
{
	Vertex& sourceVertex = graph->vertices[sourceVertexIndex];

	// FIXME: there's no guarantee that a thread that's trying to add an edge A->B will see another thread's attempt to add an edge B->A
	for (unsigned int i = 0; i < sourceVertex.numAdjacencies; i++)
	{
		if (sourceVertex.adjacencies[i] == destinationVertexIndex)
		{
			return -1;
		}
	}
	
	Vertex& destinationVertex = graph->vertices[destinationVertexIndex];

#ifdef __CUDA_ARCH__
	unsigned int mask = __ballot(1);
	unsigned int numberOfActiveThreads = __popc(mask);
	int laneId = __popc(lanemask_lt() & mask);
	int leadingThreadId = __ffs(mask) - 1;

	int oldValue;
	if (laneId == 0)
	{
		oldValue = atomicAdd((int*)&graph->numEdges, numberOfActiveThreads);

		// FIXME: checking boundaries
		if (graph->numEdges > (int)graph->maxEdges)
		{
			THROW_EXCEPTION("max. edges overflow");
		}
	}

	oldValue = __shfl(oldValue, leadingThreadId);

	int newEdgeIndex = oldValue + laneId;
#else
	int newEdgeIndex = ATOMIC_ADD(graph->numEdges, int, 1);

	// FIXME: checking boundaries
	if (graph->numEdges > (int)graph->maxEdges)
	{
		THROW_EXCEPTION("max. edges overflow");
	}
#endif

	Edge& newEdge = graph->edges[newEdgeIndex];

	newEdge.index = newEdgeIndex;
	newEdge.source = sourceVertexIndex;
	newEdge.destination = destinationVertexIndex;
	newEdge.attr1 = attr1;
	newEdge.attr2 = attr2;
	newEdge.attr3 = attr3;
	newEdge.attr4 = attr4;
	newEdge.owner = -1;

	unsigned int lastIndex = ATOMIC_ADD(sourceVertex.numOuts, unsigned int, 1);

	// FIXME: checking boundaries
	if (sourceVertex.numOuts > MAX_VERTEX_OUT_CONNECTIONS)
	{
		THROW_EXCEPTION("max. vertex connections (out) overflow");
	}

	sourceVertex.outs[lastIndex] = newEdgeIndex;

	lastIndex = ATOMIC_ADD(sourceVertex.numAdjacencies, unsigned int, 1);

	// FIXME: checking boundaries
	if (sourceVertex.numAdjacencies > MAX_VERTEX_ADJACENCIES)
	{
		THROW_EXCEPTION("max. vertex adjacencies overflow");
	}

	sourceVertex.adjacencies[lastIndex] = destinationVertexIndex;

	lastIndex = ATOMIC_ADD(destinationVertex.numIns, unsigned int, 1);

	// FIXME: checking boundaries
	if (destinationVertex.numIns > MAX_VERTEX_IN_CONNECTIONS)
	{
		THROW_EXCEPTION("max. vertex connections (in) overflow");
	}

	destinationVertex.ins[lastIndex] = newEdgeIndex;

	lastIndex = ATOMIC_ADD(destinationVertex.numAdjacencies, unsigned int, 1);

	// FIXME: checking boundaries
	if (destinationVertex.numAdjacencies > MAX_VERTEX_ADJACENCIES)
	{
		THROW_EXCEPTION("max. vertex adjacencies overflow");
	}

	destinationVertex.adjacencies[lastIndex] = sourceVertexIndex;

	insert(graph->quadtree, newEdgeIndex, Line2D(sourceVertex.getPosition(), destinationVertex.getPosition()));

	return newEdgeIndex;
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE int splitEdge(Graph* graph, Edge& edge, int splitVertexIndex)
{
	Vertex& splitVertex = graph->vertices[splitVertexIndex];
	Vertex& sourceVertex = graph->vertices[edge.source];
	Vertex& oldDestinationVertex = graph->vertices[edge.destination];

	int oldDestinationVertexIndex = edge.destination;
	edge.destination = splitVertexIndex;

/*#ifdef USE_QUADTREE
	remove(graph->quadtree, edge.index, Line2D(sourceVertex.getPosition(), oldDestinationVertex.getPosition()));
	insert(graph->quadtree, edge.index, Line2D(sourceVertex.getPosition(), splitVertex.getPosition()));
#endif*/

	replaceAdjacency(sourceVertex, oldDestinationVertexIndex, splitVertexIndex);

	unsigned int lastIndex = ATOMIC_ADD(splitVertex.numIns, unsigned int, 1);

	// FIXME: checking boundaries
	if (splitVertex.numIns >= MAX_VERTEX_IN_CONNECTIONS)
	{
		THROW_EXCEPTION("max. vertex connections (in) overflow");
	}

	splitVertex.ins[lastIndex] = edge.index;

	lastIndex = ATOMIC_ADD(splitVertex.numAdjacencies, unsigned int, 1);

	// FIXME: checking boundaries
	if (splitVertex.numAdjacencies >= MAX_VERTEX_ADJACENCIES)
	{
		THROW_EXCEPTION("max. vertex adjacencies overflow");
	}

	splitVertex.adjacencies[lastIndex] = edge.source;

	int newEdgeIndex = ATOMIC_ADD(graph->numEdges, int, 1);

	// FIXME: checking boundaries
	if (graph->numEdges >= (int)graph->maxEdges)
	{
		THROW_EXCEPTION("max. edges overflow");
	}

	Edge& newEdge = graph->edges[newEdgeIndex];

	newEdge.index = newEdgeIndex;
	newEdge.source = splitVertexIndex;
	newEdge.destination = oldDestinationVertexIndex;
	newEdge.attr1 = edge.attr1;
	newEdge.attr2 = edge.attr2;
	newEdge.attr3 = edge.attr3;
	newEdge.attr4 = edge.attr4;
	newEdge.numPrimitives = 0;
	newEdge.owner = -1;

	replaceAdjacency(oldDestinationVertex, edge.source, splitVertexIndex);
	replaceInEdge(oldDestinationVertex, edge.index, newEdgeIndex);

	lastIndex = ATOMIC_ADD(splitVertex.numOuts, unsigned int, 1);

	// FIXME: checking boundaries
	if (splitVertex.numOuts >= MAX_VERTEX_OUT_CONNECTIONS)
	{
		THROW_EXCEPTION("max. vertex connections (out) overflow");
	}

	splitVertex.outs[lastIndex] = newEdgeIndex;

	lastIndex = ATOMIC_ADD(splitVertex.numAdjacencies, unsigned int, 1);

	// FIXME: checking boundaries
	if (splitVertex.numAdjacencies >= MAX_VERTEX_ADJACENCIES)
	{
		THROW_EXCEPTION("max. vertex adjacencies overflow");
	}

	splitVertex.adjacencies[lastIndex] = oldDestinationVertexIndex;
	
	insert(graph->quadtree, newEdgeIndex, Line2D(splitVertex.getPosition(), oldDestinationVertex.getPosition()));

	return newEdgeIndex;
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE bool checkIntersection(Graph* graph, Edge& edge1, Edge& edge2, vml_vec2& intersection)
{
	// avoid intersecting parent or sibling
	if (edge1.source == edge2.source || 
		edge1.source == edge2.destination ||
		edge1.destination == edge2.source ||
		edge1.destination == edge2.destination)
	{
		return false;
	}

	vml_vec2 start1 = graph->vertices[edge1.source].getPosition();
	vml_vec2 end1 = graph->vertices[edge1.destination].getPosition();

	vml_vec2 start2 = graph->vertices[edge2.source].getPosition();
	vml_vec2 end2 = graph->vertices[edge2.destination].getPosition();

	Line2D edgeLine1(start1, end1);
	Line2D edgeLine2(start2, end2);

	if (edgeLine1.intersects(edgeLine2, intersection))
	{
		return true;
	}

#ifdef COLLECT_STATISTICS
	ATOMIC_ADD(graph->numCollisionChecks, unsigned int, 1);
#endif

	return false;
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE bool checkIntersectionWithBoundaries(Graph* graph, const Line2D& newEdgeLine, Primitive& primitive, unsigned int& start, int sourceIndex, int& edgeIndex, float& closestIntersectionDistance, vml_vec2& closestIntersection, IntersectionType& intersectionType)
{
	closestIntersectionDistance = FLT_MAX;
	intersectionType = NONE;

	for (unsigned int i = start; i < primitive.numEdges; i++)
	{
		int edgeIndex1 = primitive.edges[i];
		Edge& edge = graph->edges[edgeIndex1];

		// avoid intersecting parent or sibling
		if (edge.destination == sourceIndex || edge.source == sourceIndex)
		{
			continue;
		}

		vml_vec2 start2 = graph->vertices[edge.source].getPosition();
		vml_vec2 end2 = graph->vertices[edge.destination].getPosition();

		Line2D edgeLine(start2, end2);

		vml_vec2 intersection;
		if (newEdgeLine.intersects(edgeLine, intersection))
		{
			float distance = vml_distance(newEdgeLine.getStart(), intersection);

			if (distance < closestIntersectionDistance)
			{
				if (vml_distance(start2, intersection) <= graph->snapRadius)
				{
					if (vml_distance(newEdgeLine.getStart(), intersection) > vml_distance(newEdgeLine.getEnd(), intersection))
					{
						intersectionType = SOURCE_SOURCE_SNAPPING;
					}
					else
					{
						intersectionType = DESTINATION_SOURCE_SNAPPING;
					}

					intersection = start2;
				}

				else if (vml_distance(end2, intersection) <= graph->snapRadius)
				{
					if (vml_distance(newEdgeLine.getStart(), intersection) > vml_distance(newEdgeLine.getEnd(), intersection))
					{
						intersectionType = SOURCE_DESTINATION_SNAPPING;
					}
					else
					{
						intersectionType = DESTINATION_DESTINATION_SNAPPING;
					}

					intersection = end2;
				}

				else
				{
					intersectionType = EDGE_INTERSECTION;
				}

				closestIntersectionDistance = distance;
				closestIntersection = intersection;
				edgeIndex = edgeIndex1;
				start = i;
			}
		}

#ifdef COLLECT_STATISTICS
	ATOMIC_ADD(graph->numCollisionChecks, unsigned int, 1);
#endif
	}

	return (intersectionType != NONE);
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void addHighway(Graph* graph, int sourceIndex, const vml_vec2& direction, int& destinationIndex, vml_vec2& end)
{
	vml_vec2 start = graph->vertices[sourceIndex].getPosition();
	end = start + direction;
	destinationIndex = createVertex(graph, end);
	if (connect(graph, sourceIndex, destinationIndex, 1) == -1)
	{
		// FIXME: checking invariants
		THROW_EXCEPTION("connect(..) == -1");
	}
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE bool addStreet(Graph* graph, Primitive* primitives, int sourceIndex, const vml_vec2& direction, int boundsIndex, int& destinationIndex, vml_vec2& end)
{
	vml_vec2 start = graph->vertices[sourceIndex].getPosition();
	end = start + direction;
	Line2D newEdgeLine(start, end);

	int boundaryEdgeIndex;
	float closestIntersectionDistance;
	vml_vec2 closestIntersection;
	IntersectionType intersectionType;
	bool intersected = false;
	bool tryAgain = false;
	unsigned int i = 0;
	Primitive& bounds = primitives[boundsIndex];
	do
	{
		if (checkIntersectionWithBoundaries(graph, newEdgeLine, bounds, i, sourceIndex, boundaryEdgeIndex, closestIntersectionDistance, closestIntersection, intersectionType))
		{
			Edge& boundaryEdge = graph->edges[boundaryEdgeIndex];
			if (ATOMIC_EXCH(boundaryEdge.owner, int, THREAD_IDX_X) == -1)
			{
				end = closestIntersection;

				if (intersectionType == SOURCE_SOURCE_SNAPPING || intersectionType == SOURCE_DESTINATION_SNAPPING)
				{
					destinationIndex = boundaryEdge.source;
					connect(graph, sourceIndex, destinationIndex, 0);
				}

				else if (intersectionType == DESTINATION_SOURCE_SNAPPING)
				{
					destinationIndex = boundaryEdge.source;
					connect(graph, sourceIndex, destinationIndex, 0);
				}

				else if (intersectionType == DESTINATION_DESTINATION_SNAPPING)
				{
					destinationIndex = boundaryEdge.destination;
					connect(graph, sourceIndex, destinationIndex, 0);
				}

				else if (intersectionType == EDGE_INTERSECTION)
				{
					destinationIndex = createVertex(graph, end);
					int newEdgeIndex = splitEdge(graph, boundaryEdge, destinationIndex);
					if (connect(graph, sourceIndex, destinationIndex, 0) == -1)
					{
						// FIXME: checking invariants
						THROW_EXCEPTION("unexpected situation");
					}

					// update bounds edges and vertices
					Edge& newEdge = graph->edges[newEdgeIndex];
					newEdge.numPrimitives = boundaryEdge.numPrimitives;
					for (unsigned j = 0; j < boundaryEdge.numPrimitives; j++)
					{
						unsigned int primitiveIndex = boundaryEdge.primitives[j];
						newEdge.primitives[j] = primitiveIndex;
						Primitive& primitive = primitives[primitiveIndex];

						unsigned int lastIndex = ATOMIC_ADD(primitive.numEdges, unsigned int, 1);

						// FIXME: checking boundaries
						if (primitive.numEdges >= MAX_EDGES_PER_PRIMITIVE)
						{
							THROW_EXCEPTION("max. number of primitive edges overflow");
						}

						primitive.edges[lastIndex] = newEdgeIndex;

						lastIndex = ATOMIC_ADD(primitive.numVertices, unsigned int, 1);

						// FIXME: checking boundaries
						if (primitive.numVertices >= MAX_VERTICES_PER_PRIMITIVE)
						{
							THROW_EXCEPTION("max. number of primitive vertices overflow");
						}

						primitive.vertices[lastIndex] = destinationIndex;
					}
				}

				else
				{
					// FIXME: checking invariants
					THROW_EXCEPTION("unknown intersection type");
				}

				intersected = true;
				tryAgain = false;
				ATOMIC_EXCH(boundaryEdge.owner, int, -1);
			}
			else
			{
				tryAgain = true;
			} // atomicExch if-else
		} // check intersection if
	} while (tryAgain); // critical-section do-while

	if (intersected)
	{
		return true;
	}
	else
	{
		destinationIndex = createVertex(graph, end);
		connect(graph, sourceIndex, destinationIndex, 0);
		return false;
	}
}

#ifdef COLLECT_STATISTICS
//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getAllocatedVertices(Graph* graph)
{
	return graph->maxVertices;
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getVerticesInUse(Graph* graph)
{
	return graph->numVertices;
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getAllocatedEdges(Graph* graph)
{
	return graph->maxEdges;
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getEdgesInUse(Graph* graph)
{
	return graph->numEdges;
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getMaxVertexInConnections(Graph* graph)
{
	return MAX_VERTEX_IN_CONNECTIONS;
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getMaxVertexInConnectionsInUse(Graph* graph)
{
	unsigned int maxVerticesInConnectionsInUse = 0;

	for (int i = 0; i < graph->numVertices; i++)
	{
		Vertex& vertex = graph->vertices[i];

		if (vertex.numIns > maxVerticesInConnectionsInUse)
		{
			maxVerticesInConnectionsInUse = vertex.numIns;
		}
	}

	return maxVerticesInConnectionsInUse;
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getMaxVertexOutConnections(Graph* graph)
{
	return MAX_VERTEX_OUT_CONNECTIONS;
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getMaxVertexOutConnectionsInUse(Graph* graph)
{
	unsigned int maxVerticesOutConnectionsInUse = 0;

	for (int i = 0; i < graph->numVertices; i++)
	{
		Vertex& vertex = graph->vertices[i];

		if (vertex.numOuts > maxVerticesOutConnectionsInUse)
		{
			maxVerticesOutConnectionsInUse = vertex.numOuts;
		}
	}

	return maxVerticesOutConnectionsInUse;
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getAverageVertexInConnectionsInUse(Graph* graph)
{
	unsigned int totalVerticesInConnections = 0;

	for (int i = 0; i < graph->numVertices; i++)
	{
		Vertex& vertex = graph->vertices[i];
		totalVerticesInConnections += vertex.numIns;
	}

	return totalVerticesInConnections / graph->numVertices;
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getAverageVertexOutConnectionsInUse(Graph* graph)
{
	unsigned int totalVerticesOutConnections = 0;

	for (int i = 0; i < graph->numVertices; i++)
	{
		Vertex& vertex = graph->vertices[i];
		totalVerticesOutConnections += vertex.numIns;
	}

	return totalVerticesOutConnections / graph->numVertices;
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getAllocatedMemory(Graph* graph)
{
	unsigned int verticesBufferMemory = graph->maxVertices * sizeof(Vertex);
	unsigned int edgesBufferMemory = graph->maxEdges * sizeof(Vertex);
	return (verticesBufferMemory + edgesBufferMemory);
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getMemoryInUse(Graph* graph)
{
	unsigned int verticesBufferMemoryInUse = 0;

	for (int i = 0; i < graph->numVertices; i++)
	{
		Vertex& vertex = graph->vertices[i];
		// FIXME:
		verticesBufferMemoryInUse += vertex.numIns * sizeof(int) + vertex.numOuts * sizeof(int) + sizeof(vml_vec2) + 2 * sizeof(unsigned int) + sizeof(bool);
	}

	unsigned int edgesBufferMemoryInUse = graph->numEdges * sizeof(Vertex);
	return (verticesBufferMemoryInUse + edgesBufferMemoryInUse);
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned long getNumCollisionChecks(Graph* graph)
{
	return graph->numCollisionChecks;
}
#endif

#endif