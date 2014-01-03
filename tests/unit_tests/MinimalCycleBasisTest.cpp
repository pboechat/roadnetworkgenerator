#include <Graph.h>
#include <QuadTree.h>
#include <MinimalCycleBasis.h>
#include <Box2D.h>

#include <vector_math.h>

#include <gtest/gtest.h>

namespace RoadNetworkGraph
{

#define SNAP_RADIUS 16.0f
#define MAX_VERTICES 1000
#define MAX_EDGES 10000
#define MAX_RESULTS_PER_QUERY 10000
#define HEAP_BUFFER_SIZE 1000
#define PRIMITIVES_BUFFER_SIZE 1000
#define SEQUENCE_BUFFER_SIZE 1000
#define VISITED_BUFFER_SIZE 1000
#define QUADTREE_DEPTH 5
#define MAX_QUADRANTS 1000
#define MAX_QUADRANT_EDGES 1000

void setUpGraph(Graph& graph)
{
}

TEST(minimal_cycle_basis, extract_primitives)
{
	Box2D worldBounds(vml_vec2(0, 0), vml_vec2(512, 512));

	Graph graph;
	QuadTree quadtree;
	Vertex vertices[MAX_VERTICES];
	Edge edges[MAX_EDGES];
	EdgeIndex queryResults[MAX_RESULTS_PER_QUERY];
	Quadrant quadrants[MAX_QUADRANTS];
	QuadrantEdges quadrantEdges[MAX_QUADRANT_EDGES];

	initializeQuadtree(&quadtree, worldBounds, QUADTREE_DEPTH, MAX_RESULTS_PER_QUERY, quadrants, quadrantEdges);

	initializeGraph(&graph, SNAP_RADIUS, MAX_VERTICES, MAX_EDGES, vertices, edges, &quadtree, MAX_RESULTS_PER_QUERY, queryResults);

	setUpGraph(graph);

	allocateExtractionBuffers(HEAP_BUFFER_SIZE, PRIMITIVES_BUFFER_SIZE, SEQUENCE_BUFFER_SIZE, VISITED_BUFFER_SIZE);

	extractPrimitives(&graph);

	EXPECT_EQ(1, 1);

	freeExtractionBuffers();
}

}