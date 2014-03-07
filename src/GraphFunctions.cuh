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
#include <IntersectionType.h>
#ifdef USE_CUDA
#include <cutil.h>
#endif

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
#if defined(__CUDA_ARCH__) && (CUDA_CC >= 30)
	unsigned int mask = __ballot(1);
	unsigned int numberOfActiveThreads = __popc(mask);
	int laneId = __popc(lanemask_lt() & mask);
	int leadingThreadId = __ffs(mask) - 1;

	int oldValue;
	if (laneId == 0)
	{
		oldValue = atomicAdd((int*)&graph->numVertices, numberOfActiveThreads);

		// FIXME: checking boundaries
		if (graph->numVertices > (int)graph->maxVertices)
		{
			THROW_EXCEPTION1("max. vertices overflow (%d)", graph->numVertices);
		}
	}

	oldValue = __shfl(oldValue, leadingThreadId);

	newVertexIndex = oldValue + laneId;
#else
	newVertexIndex = ATOMIC_ADD(graph->numVertices, int, 1);

	// FIXME: checking boundaries
	if (graph->numVertices > (int)graph->maxVertices)
	{
		THROW_EXCEPTION1("max. vertices overflow (%d)", graph->numVertices);
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
		THROW_EXCEPTION1("max. vertices overflow (%d)", graph->numVertices);
	}

	int newVertexIndex = ATOMIC_ADD(graph->numVertices, int, 1);

	Vertex& newVertex = graph->vertices[newVertexIndex];

	newVertex.index = newVertexIndex;
	newVertex.setPosition(position);

	return newVertexIndex;
}
#endif

#if defined(USE_CUDA) && (CUDA_CC >= 30)
//////////////////////////////////////////////////////////////////////////
__device__ int connect(Graph* graph, int sourceVertexIndex, int destinationVertexIndex, bool updateQuadtree = true, char attr1 = 0, char attr2 = 0, char attr3 = 0, char attr4 = 0)
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
			THROW_EXCEPTION1("max. edges overflow (%d)", graph->numEdges);
		}
	}

	oldValue = __shfl(oldValue, leadingThreadId);

	unsigned int newEdgeIndex = oldValue + laneId;

	Edge& newEdge = graph->edges[newEdgeIndex];

	newEdge.index = newEdgeIndex;
	newEdge.source = sourceVertexIndex;
	newEdge.destination = destinationVertexIndex;
	newEdge.attr1 = attr1;
	newEdge.attr2 = attr2;
	newEdge.attr3 = attr3;
	newEdge.attr4 = attr4;
	newEdge.owner = -1;

	unsigned int lastIndex = atomicAdd((unsigned int*)&sourceVertex.numOuts, 1);

	// FIXME: checking boundaries
	if (sourceVertex.numOuts > MAX_VERTEX_OUT_CONNECTIONS)
	{
		THROW_EXCEPTION1("max. vertex connections (out) overflow (%d)", sourceVertex.numOuts);
	}

	sourceVertex.outs[lastIndex] = newEdgeIndex;

	lastIndex = atomicAdd((unsigned int*)&sourceVertex.numAdjacencies, 1);

	// FIXME: checking boundaries
	if (sourceVertex.numAdjacencies > MAX_VERTEX_ADJACENCIES)
	{
		THROW_EXCEPTION1("max. vertex adjacencies overflow (%d)", sourceVertex.numAdjacencies);
	}

	sourceVertex.adjacencies[lastIndex] = destinationVertexIndex;

	lastIndex = atomicAdd((unsigned int*)&destinationVertex.numIns, 1);

	// FIXME: checking boundaries
	if (destinationVertex.numIns > MAX_VERTEX_IN_CONNECTIONS)
	{
		THROW_EXCEPTION1("max. vertex connections (in) overflow (%d)", destinationVertex.numIns);
	}

	destinationVertex.ins[lastIndex] = newEdgeIndex;

	lastIndex = atomicAdd((unsigned int*)&destinationVertex.numAdjacencies, 1);

	// FIXME: checking boundaries
	if (destinationVertex.numAdjacencies > MAX_VERTEX_ADJACENCIES)
	{
		THROW_EXCEPTION1("max. vertex adjacencies overflow (%d)", destinationVertex.numAdjacencies);
	}

	destinationVertex.adjacencies[lastIndex] = sourceVertexIndex;

	if (updateQuadtree)
	{
		insert(graph->quadtree, newEdgeIndex, Line2D(sourceVertex.getPosition(), destinationVertex.getPosition()));
	}

	return newEdgeIndex;
}
#else
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE int connect(Graph* graph, int sourceVertexIndex, int destinationVertexIndex, bool updateQuadtree = true, char attr1 = 0, char attr2 = 0, char attr3 = 0, char attr4 = 0)
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

	int newEdgeIndex = ATOMIC_ADD(graph->numEdges, int, 1);

	// FIXME: checking boundaries
	if (graph->numEdges > (int)graph->maxEdges)
	{
		THROW_EXCEPTION1("max. edges overflow (%d)", graph->numEdges);
	}

	Edge& newEdge = graph->edges[newEdgeIndex];

	newEdge.index = newEdgeIndex;
	newEdge.source = sourceVertexIndex;
	newEdge.destination = destinationVertexIndex;
	newEdge.attr1 = attr1;
	newEdge.attr2 = attr2;
	newEdge.attr3 = attr3;
	newEdge.attr4 = attr4;
	newEdge.numPrimitives = 0;
	newEdge.owner = -1;

	unsigned int lastIndex = ATOMIC_ADD(sourceVertex.numOuts, unsigned int, 1);

	// FIXME: checking boundaries
	if (sourceVertex.numOuts > MAX_VERTEX_OUT_CONNECTIONS)
	{
		THROW_EXCEPTION1("max. vertex connections (out) overflow (%d)", sourceVertex.numOuts);
	}

	sourceVertex.outs[lastIndex] = newEdgeIndex;

	lastIndex = ATOMIC_ADD(sourceVertex.numAdjacencies, unsigned int, 1);

	// FIXME: checking boundaries
	if (sourceVertex.numAdjacencies > MAX_VERTEX_ADJACENCIES)
	{
		THROW_EXCEPTION1("max. vertex adjacencies overflow (%d)", sourceVertex.numAdjacencies);
	}

	sourceVertex.adjacencies[lastIndex] = destinationVertexIndex;

	lastIndex = ATOMIC_ADD(destinationVertex.numIns, unsigned int, 1);

	// FIXME: checking boundaries
	if (destinationVertex.numIns > MAX_VERTEX_IN_CONNECTIONS)
	{
		THROW_EXCEPTION1("max. vertex connections (in) overflow (%d)", destinationVertex.numIns);
	}

	destinationVertex.ins[lastIndex] = newEdgeIndex;

	lastIndex = ATOMIC_ADD(destinationVertex.numAdjacencies, unsigned int, 1);

	// FIXME: checking boundaries
	if (destinationVertex.numAdjacencies > MAX_VERTEX_ADJACENCIES)
	{
		THROW_EXCEPTION1("max. vertex adjacencies overflow (%d)", destinationVertex.numAdjacencies);
	}

	destinationVertex.adjacencies[lastIndex] = sourceVertexIndex;

	if (updateQuadtree)
	{
		insert(graph->quadtree, newEdgeIndex, Line2D(sourceVertex.getPosition(), destinationVertex.getPosition()));
	}

	return newEdgeIndex;
}
#endif

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE int splitEdge(Graph* graph, Edge& edge, int splitVertexIndex, bool updateQuadtree = true)
{
	Vertex& splitVertex = graph->vertices[splitVertexIndex];
	Vertex& sourceVertex = graph->vertices[edge.source];
	Vertex& oldDestinationVertex = graph->vertices[edge.destination];

	int oldDestinationVertexIndex = edge.destination;
	edge.destination = splitVertexIndex;

	replaceAdjacency(sourceVertex, oldDestinationVertexIndex, splitVertexIndex);

	unsigned int lastIndex = ATOMIC_ADD(splitVertex.numIns, unsigned int, 1);

	// FIXME: checking boundaries
	if (splitVertex.numIns > MAX_VERTEX_IN_CONNECTIONS)
	{
		THROW_EXCEPTION1("max. vertex connections (in) overflow (%d)", splitVertex.numIns);
	}

	splitVertex.ins[lastIndex] = edge.index;

	lastIndex = ATOMIC_ADD(splitVertex.numAdjacencies, unsigned int, 1);

	// FIXME: checking boundaries
	if (splitVertex.numAdjacencies > MAX_VERTEX_ADJACENCIES)
	{
		THROW_EXCEPTION1("max. vertex adjacencies overflow (%d)", splitVertex.numAdjacencies);
	}

	splitVertex.adjacencies[lastIndex] = edge.source;

	int newEdgeIndex = ATOMIC_ADD(graph->numEdges, int, 1);

	// FIXME: checking boundaries
	if (graph->numEdges > (int)graph->maxEdges)
	{
		THROW_EXCEPTION1("max. edges overflow (%d)", graph->numEdges);
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
	if (splitVertex.numOuts > MAX_VERTEX_OUT_CONNECTIONS)
	{
		THROW_EXCEPTION1("max. vertex connections (out) overflow (%d)", splitVertex.numOuts);
	}

	splitVertex.outs[lastIndex] = newEdgeIndex;

	lastIndex = ATOMIC_ADD(splitVertex.numAdjacencies, unsigned int, 1);

	// FIXME: checking boundaries
	if (splitVertex.numAdjacencies > MAX_VERTEX_ADJACENCIES)
	{
		THROW_EXCEPTION1("max. vertex adjacencies overflow (%d)", splitVertex.numAdjacencies);
	}

	splitVertex.adjacencies[lastIndex] = oldDestinationVertexIndex;
	
	if (updateQuadtree)
	{
		insert(graph->quadtree, newEdgeIndex, Line2D(splitVertex.getPosition(), oldDestinationVertex.getPosition()));
	}

	return newEdgeIndex;
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE bool checkIntersection(Graph* graph, const Edge& edge1, const Edge& edge2, vml_vec2& intersection)
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
DEVICE_CODE bool checkIntersection(Graph* graph, const Line2D& edgeLine1, const Edge& edge2, vml_vec2& intersection)
{
	vml_vec2 start2 = graph->vertices[edge2.source].getPosition();
	vml_vec2 end2 = graph->vertices[edge2.destination].getPosition();

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
DEVICE_CODE void addHighway(Graph* graph, int sourceIndex, const vml_vec2& direction, int& destinationIndex, vml_vec2& end)
{
	vml_vec2 start = graph->vertices[sourceIndex].getPosition();
	end = start + direction;
	destinationIndex = createVertex(graph, end);
	if (connect(graph, sourceIndex, destinationIndex, true, 1) == -1)
	{
		// FIXME: checking invariants
		THROW_EXCEPTION("connect(..) == -1");
	}
}


//////////////////////////////////////////////////////////////////////////
DEVICE_CODE bool findStreetConnection(Graph* graph, int sourceIndex, char childCode, int& destinationIndex)
{
	if (childCode == UP_CHILD)
	{
		// source is spawn point
		if (graph->vertices[sourceIndex].numIns != 1)
		{
			return false;
		}

		Edge& parent = graph->edges[graph->vertices[sourceIndex].ins[0]];

		// source is up child
		if (parent.attr2 == 2)
		{
			return false;
		}

		Vertex& parentSource = graph->vertices[parent.source];

		bool foundUncle = false;
		int uncleDestinationIndex;
		for (unsigned int i = 0; i < parentSource.numOuts; i++)
		{
			Edge& edge = graph->edges[parentSource.outs[i]];

			if (edge.attr2 == 2)
			{
				uncleDestinationIndex = edge.destination;
				foundUncle = true;
				break;
			}
		}

		// FIXME: checking invariants
		if (!foundUncle)
		{
			THROW_EXCEPTION("!foundUncle");
		}

		Vertex& uncleDestination = graph->vertices[uncleDestinationIndex];

		for (unsigned int i = 0; i < uncleDestination.numOuts; i++)
		{
			Edge& edge = graph->edges[uncleDestination.outs[i]];

			if (edge.attr2 == 1)
			{
				destinationIndex = edge.destination;
				return true;
			}
		}
	}

	return false;
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE bool addStreet(Graph* graph, Primitive* primitives, int sourceIndex, const vml_vec2& direction, unsigned int boundsIndex, char childCode, int& destinationIndex, vml_vec2& end)
{
	vml_vec2 start = graph->vertices[sourceIndex].getPosition();
	end = start + direction;
	Line2D newEdgeLine(start, end);

	bool intersected = false;
	Primitive& bounds = primitives[boundsIndex];
	
	for (unsigned int i = 0; i < bounds.numEdges; i++)
	{
		Edge& boundaryEdge = graph->edges[bounds.edges[i]];

		bool tryAgain;
		do
		{
			tryAgain = true;
			if (ATOMIC_EXCH(boundaryEdge.owner, int, THREAD_IDX_X) == -1)
			{
				vml_vec2 intersection;
				if (checkIntersection(graph, newEdgeLine, boundaryEdge, intersection))
				{
					destinationIndex = createVertex(graph, intersection);
					int newEdgeIndex = splitEdge(graph, boundaryEdge, destinationIndex);
					if (connect(graph, sourceIndex, destinationIndex, false, 0, childCode) == -1)
					{
						// FIXME: checking invariants
						THROW_EXCEPTION("unexpected situation");
					}

					// update bounds edges and vertices
					Edge& newEdge = graph->edges[newEdgeIndex];
					newEdge.numPrimitives = boundaryEdge.numPrimitives;
					for (unsigned int j = 0; j < boundaryEdge.numPrimitives; j++)
					{
						unsigned int primitiveIndex = boundaryEdge.primitives[j];
						newEdge.primitives[j] = primitiveIndex;
						Primitive& primitive = primitives[primitiveIndex];

						unsigned int lastEdgeIndex = ATOMIC_ADD(primitive.numEdges, unsigned int, 1);

						// FIXME: checking boundaries
						if (primitive.numEdges > MAX_EDGES_PER_PRIMITIVE)
						{
							THROW_EXCEPTION1("max. number of primitive edges overflow (%d)", primitive.numEdges);
						}

						primitive.edges[lastEdgeIndex] = newEdgeIndex;

						unsigned int lastVertexIndex = ATOMIC_ADD(primitive.numVertices, unsigned int, 1);

						// FIXME: checking boundaries
						if (primitive.numVertices > MAX_VERTICES_PER_PRIMITIVE)
						{
							THROW_EXCEPTION1("max. number of primitive vertices overflow (%d)", primitive.numVertices);
						}

						primitive.vertices[lastVertexIndex] = intersection;
					}

					intersected = true;
				}
				
				tryAgain = false;
				ATOMIC_EXCH(boundaryEdge.owner, int, -1);
			} // critical-section if

#ifdef COLLECT_STATISTICS
			ATOMIC_ADD(graph->numCollisionChecks, unsigned int, 1);
#endif
		}  while (tryAgain);

		if (intersected)
		{
			break;
		}
	} // for loop

	if (!intersected)
	{
		if (!findStreetConnection(graph, sourceIndex, childCode, destinationIndex))
		{
			destinationIndex = createVertex(graph, end);
		}

		if (connect(graph, sourceIndex, destinationIndex, false, 0, childCode) == -1)
		{
			THROW_EXCEPTION("unexpected situation");
		}

		return false;
	}

	return true;
}

#ifdef COLLECT_STATISTICS
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
HOST_CODE float getAverageVertexInConnectionsInUse(Graph* graph)
{
	unsigned int totalVerticesInConnections = 0;

	for (int i = 0; i < graph->numVertices; i++)
	{
		Vertex& vertex = graph->vertices[i];
		totalVerticesInConnections += vertex.numIns;
	}

	return totalVerticesInConnections / (float)graph->numVertices;
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE float getAverageVertexOutConnectionsInUse(Graph* graph)
{
	unsigned int totalVerticesOutConnections = 0;

	for (int i = 0; i < graph->numVertices; i++)
	{
		Vertex& vertex = graph->vertices[i];
		totalVerticesOutConnections += vertex.numOuts;
	}

	return totalVerticesOutConnections / (float)graph->numVertices;
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getAllocatedMemory(Graph* graph)
{
	unsigned int verticesMemory = graph->maxVertices * sizeof(Vertex);
	unsigned int edgesMemory = graph->maxEdges * sizeof(Vertex);
	return sizeof(Graph) + (verticesMemory + edgesMemory);
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int getMemoryInUse(Graph* graph)
{
	unsigned int verticesMemoryInUse = 0;

	for (int i = 0; i < graph->numVertices; i++)
	{
		Vertex& vertex = graph->vertices[i];
		verticesMemoryInUse += vertex.numIns * sizeof(int) + vertex.numOuts * sizeof(int) + sizeof(vml_vec2) + 2 * sizeof(unsigned int) + sizeof(bool);
	}

	unsigned int edgesMemoryInUse = graph->numEdges * sizeof(Vertex);
	return sizeof(Graph) + (verticesMemoryInUse + edgesMemoryInUse);
}

#endif

#endif