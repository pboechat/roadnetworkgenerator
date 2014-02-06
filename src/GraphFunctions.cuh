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
#ifdef USE_QUADTREE
#include <QuadTree.h>
#include <QueryResults.h>
#include <QuadTreeFunctions.cuh>
#endif
#include <VertexFunctions.cuh>
#include <Primitive.h>
#ifdef USE_CUDA
#include <cutil.h>
#endif

//////////////////////////////////////////////////////////////////////////
enum IntersectionType
{
	NONE,
	SOURCE,
	DESTINATION,
	EDGE
};

#ifdef USE_QUADTREE
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
#else
//////////////////////////////////////////////////////////////////////////
GLOBAL_CODE void initializeGraphOnDevice(Graph* graph, float snapRadius, unsigned int maxVertices, unsigned int maxEdges, Vertex* vertices, Edge* edges)
{
	graph->numVertices = 0;
	graph->numEdges = 0;
	graph->vertices = vertices;
	graph->edges = edges;
	graph->maxVertices = maxVertices;
	graph->maxEdges = maxEdges;
	graph->snapRadius = snapRadius;
#ifdef COLLECT_STATISTICS
	graph->numCollisionChecks = 0;
#endif
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE void initializeGraphOnHost(Graph* graph, float snapRadius, unsigned int maxVertices, unsigned int maxEdges, Vertex* vertices, Edge* edges)
{
	graph->numVertices = 0;
	graph->numEdges = 0;
	graph->vertices = vertices;
	graph->edges = edges;
	graph->maxVertices = maxVertices;
	graph->maxEdges = maxEdges;
	graph->snapRadius = snapRadius;
#ifdef COLLECT_STATISTICS
	graph->numCollisionChecks = 0;
#endif
}
#endif

//////////////////////////////////////////////////////////////////////////
GLOBAL_CODE void updateNonPointerFields(Graph* graph, unsigned int numVertices, unsigned int numEdges, unsigned int maxVertices, unsigned int maxEdges
#ifdef COLLECT_STATISTICS
	, unsigned long numCollisionChecks
#endif	
)
{
	graph->numVertices = numVertices;
	graph->numEdges = numEdges;
	graph->maxVertices = maxVertices;
	graph->maxEdges = maxEdges;
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

#if defined(USE_CUDA) && (CUDA_CC >= 30)
//////////////////////////////////////////////////////////////////////////
__device__ int connect(Graph* graph, int sourceVertexIndex, int destinationVertexIndex, char attr1 = 0, char attr2 = 0, char attr3 = 0, char attr4 = 0)
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
		if (graph->numEdges >= (int)graph->maxEdges)
		{
			THROW_EXCEPTION("max. edges overflow");
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
	if (sourceVertex.numOuts >= MAX_VERTEX_OUT_CONNECTIONS)
	{
		THROW_EXCEPTION("max. vertex connections (out) overflow");
	}

	sourceVertex.outs[lastIndex] = newEdgeIndex;

	lastIndex = atomicAdd((unsigned int*)&sourceVertex.numAdjacencies, 1);

	// FIXME: checking boundaries
	if (sourceVertex.numAdjacencies >= MAX_VERTEX_ADJACENCIES)
	{
		THROW_EXCEPTION("max. vertex adjacencies overflow");
	}

	sourceVertex.adjacencies[lastIndex] = destinationVertexIndex;

	lastIndex = atomicAdd((unsigned int*)&destinationVertex.numIns, 1);

	// FIXME: checking boundaries
	if (destinationVertex.numIns >= MAX_VERTEX_IN_CONNECTIONS)
	{
		THROW_EXCEPTION("max. vertex connections (in) overflow");
	}

	destinationVertex.ins[lastIndex] = newEdgeIndex;

	lastIndex = atomicAdd((unsigned int*)&destinationVertex.numAdjacencies, 1);

	// FIXME: checking boundaries
	if (destinationVertex.numAdjacencies >= MAX_VERTEX_ADJACENCIES)
	{
		THROW_EXCEPTION("max. vertex adjacencies overflow");
	}

	destinationVertex.adjacencies[lastIndex] = sourceVertexIndex;

#ifdef USE_QUADTREE
	insert(graph->quadtree, newEdgeIndex, Line2D(sourceVertex.getPosition(), destinationVertex.getPosition()));
#endif

	return newEdgeIndex;
}
#else
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE int connect(Graph* graph, int sourceVertexIndex, int destinationVertexIndex, char attr1 = 0, char attr2 = 0, char attr3 = 0, char attr4 = 0)
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
	if (graph->numEdges >= (int)graph->maxEdges)
	{
		THROW_EXCEPTION("max. edges overflow");
	}

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
	if (sourceVertex.numOuts >= MAX_VERTEX_OUT_CONNECTIONS)
	{
		THROW_EXCEPTION("max. vertex connections (out) overflow");
	}

	sourceVertex.outs[lastIndex] = newEdgeIndex;

	lastIndex = ATOMIC_ADD(sourceVertex.numAdjacencies, unsigned int, 1);

	// FIXME: checking boundaries
	if (sourceVertex.numAdjacencies >= MAX_VERTEX_ADJACENCIES)
	{
		THROW_EXCEPTION("max. vertex adjacencies overflow");
	}

	sourceVertex.adjacencies[lastIndex] = destinationVertexIndex;

	lastIndex = ATOMIC_ADD(destinationVertex.numIns, unsigned int, 1);

	// FIXME: checking boundaries
	if (destinationVertex.numIns >= MAX_VERTEX_IN_CONNECTIONS)
	{
		THROW_EXCEPTION("max. vertex connections (in) overflow");
	}

	destinationVertex.ins[lastIndex] = newEdgeIndex;

	lastIndex = ATOMIC_ADD(destinationVertex.numAdjacencies, unsigned int, 1);

	// FIXME: checking boundaries
	if (destinationVertex.numAdjacencies >= MAX_VERTEX_ADJACENCIES)
	{
		THROW_EXCEPTION("max. vertex adjacencies overflow");
	}

	destinationVertex.adjacencies[lastIndex] = sourceVertexIndex;

#ifdef USE_QUADTREE
	insert(graph->quadtree, newEdgeIndex, Line2D(sourceVertex.getPosition(), destinationVertex.getPosition()));
#endif

	return newEdgeIndex;
}
#endif

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
	
#ifdef USE_QUADTREE
	insert(graph->quadtree, newEdgeIndex, Line2D(splitVertex.getPosition(), oldDestinationVertex.getPosition()));
#endif

	return newEdgeIndex;
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE bool checkIntersection(Graph* graph, const Line2D& newEdgeLine, int& start, int sourceIndex, int& edgeIndex, float& closestIntersectionDistance, vml_vec2& closestIntersection, IntersectionType& intersectionType)
{
	closestIntersectionDistance = FLT_MAX;
	intersectionType = NONE;

	for (int i = start; i < graph->numEdges; i++)
	{
		Edge& edge = graph->edges[i];

		// avoid intersecting parent or sibling
		if (edge.destination == sourceIndex || edge.source == sourceIndex)
		{
			continue;
		}

		vml_vec2 sourceVertexPosition = graph->vertices[edge.source].getPosition();
		vml_vec2 destinationVertexPosition = graph->vertices[edge.destination].getPosition();

		Line2D edgeLine(sourceVertexPosition, destinationVertexPosition);

		vml_vec2 intersection;
		if (newEdgeLine.intersects(edgeLine, intersection))
		{
			float distance = vml_distance(newEdgeLine.getStart(), intersection);

			if (distance < closestIntersectionDistance)
			{
				if (vml_distance(sourceVertexPosition, intersection) <= graph->snapRadius)
				{
					intersection = sourceVertexPosition;
					intersectionType = SOURCE;
				}

				else if (vml_distance(destinationVertexPosition, intersection) <= graph->snapRadius)
				{
					intersection = destinationVertexPosition;
					intersectionType = DESTINATION;
				}

				else
				{
					intersectionType = EDGE;
				}

				closestIntersectionDistance = distance;
				closestIntersection = intersection;
				edgeIndex = i;
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
DEVICE_CODE bool checkIntersection(Graph* graph, const Line2D& newEdgeLine, int* edgesList, unsigned int edgesListSize, unsigned int& start, int sourceIndex, int& edgeIndex, float& closestIntersectionDistance, vml_vec2& closestIntersection, IntersectionType& intersectionType)
{
	closestIntersectionDistance = FLT_MAX;
	intersectionType = NONE;

	for (unsigned int i = start; i < edgesListSize; i++)
	{
		int edgeIndex1 = edgesList[i];
		Edge& edge = graph->edges[edgeIndex1];

		// avoid intersecting parent or sibling
		if (edge.destination == sourceIndex || edge.source == sourceIndex)
		{
			continue;
		}

		vml_vec2 sourceVertexPosition = graph->vertices[edge.source].getPosition();
		vml_vec2 destinationVertexPosition = graph->vertices[edge.destination].getPosition();

		Line2D edgeLine(sourceVertexPosition, destinationVertexPosition);

		vml_vec2 intersection;
		if (newEdgeLine.intersects(edgeLine, intersection))
		{
			float distance = vml_distance(newEdgeLine.getStart(), intersection);

			if (distance < closestIntersectionDistance)
			{
				if (vml_distance(sourceVertexPosition, intersection) <= graph->snapRadius)
				{
					intersection = sourceVertexPosition;
					intersectionType = SOURCE;
				}

				else if (vml_distance(destinationVertexPosition, intersection) <= graph->snapRadius)
				{
					intersection = destinationVertexPosition;
					intersectionType = DESTINATION;
				}

				else
				{
					intersectionType = EDGE;
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

#ifdef USE_QUADTREE
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE bool checkIntersection(Graph* graph, const Line2D& newEdgeLine, QueryResults& queryResults, unsigned int& start1, unsigned int& start2, int sourceIndex, int& edgeIndex, float& closestIntersectionDistance, vml_vec2& closestIntersection, IntersectionType& intersectionType)
{
	closestIntersectionDistance = FLT_MAX;
	intersectionType = NONE;

	for (unsigned int i = start1; i < queryResults.numResults; i++)
	{
		QuadrantEdges& quadrantEdges = graph->quadtree->quadrantsEdges[queryResults.results[i]];
		for (unsigned int j = start2; j < quadrantEdges.lastEdgeIndex; j++)
		{
			int edgeIndex1 = quadrantEdges.edges[j];
			Edge& edge = graph->edges[edgeIndex1];

			// avoid intersecting parent or sibling
			if (edge.destination == sourceIndex || edge.source == sourceIndex)
			{
				continue;
			}

			vml_vec2 sourceVertexPosition = graph->vertices[edge.source].getPosition();
			vml_vec2 destinationVertexPosition = graph->vertices[edge.destination].getPosition();

			Line2D edgeLine(sourceVertexPosition, destinationVertexPosition);

			vml_vec2 intersection;
			if (newEdgeLine.intersects(edgeLine, intersection))
			{
				float distance = vml_distance(newEdgeLine.getStart(), intersection);

				if (distance < closestIntersectionDistance)
				{
					if (vml_distance(sourceVertexPosition, intersection) <= graph->snapRadius)
					{
						intersection = sourceVertexPosition;
						intersectionType = SOURCE;
					}

					else if (vml_distance(destinationVertexPosition, intersection) <= graph->snapRadius)
					{
						intersection = destinationVertexPosition;
						intersectionType = DESTINATION;
					}

					else
					{
						intersectionType = EDGE;
					}

					closestIntersectionDistance = distance;
					closestIntersection = intersection;
					edgeIndex = edgeIndex1;
					start1 = i;
					start2 = j;
				}
			}

#ifdef COLLECT_STATISTICS
			ATOMIC_ADD(graph->numCollisionChecks, unsigned int, 1);
#endif
		}
	}

	return (intersectionType != NONE);
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE bool addHighway(Graph* graph, int sourceIndex, const vml_vec2& direction, int& destinationIndex, vml_vec2& end)
{
	vml_vec2 start = graph->vertices[sourceIndex].getPosition();
	end = start + direction;
	Line2D newEdgeLine(start, end);

	QueryResults queryResults;
	query(graph->quadtree, newEdgeLine, queryResults);

	int edgeIndex;
	float closestIntersectionDistance;
	vml_vec2 closestIntersection;
	IntersectionType intersectionType;
	bool tryAgain = false;
	bool intersected = false;
	unsigned int i = 0;
	unsigned int j = 0;
	do
	{
		if (checkIntersection(graph, newEdgeLine, queryResults, i, j, sourceIndex, edgeIndex, closestIntersectionDistance, closestIntersection, intersectionType))
		{
			Edge& edge = graph->edges[edgeIndex];
			if (ATOMIC_EXCH(edge.owner, int, THREAD_IDX_X) == -1)
			{
				end = closestIntersection;

				if (intersectionType == SOURCE)
				{
					destinationIndex = edge.source;
					connect(graph, sourceIndex, destinationIndex, 1);
				}

				else if (intersectionType == DESTINATION)
				{
					destinationIndex = edge.destination;
					connect(graph, sourceIndex, destinationIndex, 1);
				}

				else if (intersectionType == EDGE)
				{
					destinationIndex = createVertex(graph, end);
					splitEdge(graph, edge, destinationIndex);
					if (connect(graph, sourceIndex, destinationIndex, 1) == -1)
					{
						// FIXME: checking invariants
						THROW_EXCEPTION("unexpected situation");
					}
				}

				else
				{
					// FIXME: checking invariants
					THROW_EXCEPTION("unknown intersection type");
				}

				intersected = true;
				tryAgain = false;
				ATOMIC_EXCH(edge.owner, int, -1);
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
		connect(graph, sourceIndex, destinationIndex, 1);
		return false;
	}
}
#else
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE bool addHighway(Graph* graph, int sourceIndex, const vml_vec2& direction, int& destinationIndex, vml_vec2& end)
{
	vml_vec2 start = graph->vertices[sourceIndex].getPosition();
	end = start + direction;
	Line2D newEdgeLine(start, end);

	int edgeIndex;
	float closestIntersectionDistance;
	vml_vec2 closestIntersection;
	IntersectionType intersectionType;
	bool intersected = false;
	bool tryAgain = false;
	int i = 0;
	do
	{
		if (checkIntersection(graph, newEdgeLine, i, sourceIndex, edgeIndex, closestIntersectionDistance, closestIntersection, intersectionType))
		{
			Edge& edge = graph->edges[edgeIndex];
			if (ATOMIC_EXCH(edge.owner, int, THREAD_IDX_X) == -1)
			{
				end = closestIntersection;

				if (intersectionType == SOURCE)
				{
					destinationIndex = edge.source;
					connect(graph, sourceIndex, destinationIndex, 1);
				}

				else if (intersectionType == DESTINATION)
				{
					destinationIndex = edge.destination;
					connect(graph, sourceIndex, destinationIndex, 1);
				}

				else if (intersectionType == EDGE)
				{
					destinationIndex = createVertex(graph, end);
					splitEdge(graph, edge, destinationIndex);
					if (connect(graph, sourceIndex, destinationIndex, 1) == -1)
					{
						// FIXME: checking invariants
						THROW_EXCEPTION("unexpected situation");
					}
				}

				else
				{
					// FIXME: checking invariants
					THROW_EXCEPTION("unknown intersection type");
				}

				intersected = true;
				tryAgain = false;
				ATOMIC_EXCH(edge.owner, int, -1);
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
		connect(graph, sourceIndex, destinationIndex, 1);
		return false;
	}
}
#endif

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE bool addStreet(Graph* graph, Primitive* primitives, int sourceIndex, const vml_vec2& direction, int boundsIndex, int& destinationIndex, vml_vec2& end)
{
	vml_vec2 start = graph->vertices[sourceIndex].getPosition();
	end = start + direction;
	Line2D newEdgeLine(start, end);
	
	int edgeIndex;
	float closestIntersectionDistance;
	vml_vec2 closestIntersection;
	IntersectionType intersectionType;
	bool intersected = false;
	bool tryAgain = false;
	unsigned int i = 0;
	Primitive& bounds = primitives[boundsIndex];
	do
	{
		if (checkIntersection(graph, newEdgeLine, (int*)bounds.edges, (unsigned int)bounds.numEdges, i, sourceIndex, edgeIndex, closestIntersectionDistance, closestIntersection, intersectionType))
		{
			Edge& edge = graph->edges[edgeIndex];
			if (ATOMIC_EXCH(edge.owner, int, THREAD_IDX_X) == -1)
			{
				end = closestIntersection;

				if (intersectionType == SOURCE)
				{
					destinationIndex = edge.source;
					connect(graph, sourceIndex, destinationIndex, 0);
				}

				else if (intersectionType == DESTINATION)
				{
					destinationIndex = edge.destination;
					connect(graph, sourceIndex, destinationIndex, 0);
				}

				else if (intersectionType == EDGE)
				{
					destinationIndex = createVertex(graph, end);
					int newEdgeIndex = splitEdge(graph, edge, destinationIndex);
					if (connect(graph, sourceIndex, destinationIndex, 0) == -1)
					{
						// FIXME: checking invariants
						THROW_EXCEPTION("unexpected situation");
					}

					// update bounds edges and vertices
					Edge& newEdge = graph->edges[newEdgeIndex];
					newEdge.numPrimitives = edge.numPrimitives;
					for (unsigned j = 0; j < edge.numPrimitives; j++)
					{
						unsigned int primitiveIndex = edge.primitives[j];
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

						primitive.vertices[lastIndex] = end;
					}
				}

				else
				{
					// FIXME: checking invariants
					THROW_EXCEPTION("unknown intersection type");
				}

				intersected = true;
				tryAgain = false;
				ATOMIC_EXCH(edge.owner, int, -1);
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