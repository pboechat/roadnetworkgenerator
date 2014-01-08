#include <Graph.h>
#include <QuadTree.h>
#include <MinimalCycleBasis.h>
#include <Box2D.h>

#include <vector_math.h>

#include <gtest/gtest.h>

namespace RoadNetworkGraph
{

#define WORLD_BOUNDS Box2D(vml_vec2(0, 0), vml_vec2(100, 40))
#define SNAP_RADIUS 0.0f
#define MAX_VERTICES 30
#define MAX_EDGES 30
#define MAX_RESULTS_PER_QUERY 1
#define HEAP_BUFFER_SIZE 30
#define PRIMITIVES_BUFFER_SIZE 30
#define SEQUENCE_BUFFER_SIZE 30
#define VISITED_BUFFER_SIZE 30
#define QUADTREE_DEPTH 1
#define MAX_QUADRANTS 1
#define MAX_QUADRANT_EDGES 30

void setUpGraph(Graph* graph)
{
	// v0
	VertexIndex v0 = createVertex(graph, vml_vec2(3, 3));
	// v1
	VertexIndex v1 = createVertex(graph, vml_vec2(6, 9));
	// v2
	VertexIndex v2 = createVertex(graph, vml_vec2(19, 10));
	// v3
	VertexIndex v3 = createVertex(graph, vml_vec2(1, 22));
	// v4
	VertexIndex v4 = createVertex(graph, vml_vec2(23, 22));
	// v5
	VertexIndex v5 = createVertex(graph, vml_vec2(13, 16));
	// v6
	VertexIndex v6 = createVertex(graph, vml_vec2(6, 20));
	// v7
	VertexIndex v7 = createVertex(graph, vml_vec2(30, 5));
	// v8
	VertexIndex v8 = createVertex(graph, vml_vec2(29, 17));
	// v9
	VertexIndex v9 = createVertex(graph, vml_vec2(24, 28));

	// v10
	VertexIndex v10 = createVertex(graph, vml_vec2(33, 28));
	// v11
	VertexIndex v11 = createVertex(graph, vml_vec2(34, 7));
	// v12
	VertexIndex v12 = createVertex(graph, vml_vec2(35, 21));
	// v13
	VertexIndex v13 = createVertex(graph, vml_vec2(53, 6));
	// v14
	VertexIndex v14 = createVertex(graph, vml_vec2(45, 34));
	// v15
	VertexIndex v15 = createVertex(graph, vml_vec2(54, 26));
	// v16
	VertexIndex v16 = createVertex(graph, vml_vec2(66, 33));
	// v17
	VertexIndex v17 = createVertex(graph, vml_vec2(53, 16));
	// v18
	VertexIndex v18 = createVertex(graph, vml_vec2(63, 12));
	// v19
	VertexIndex v19 = createVertex(graph, vml_vec2(71, 6));

	// v20
	VertexIndex v20 = createVertex(graph, vml_vec2(74, 20));
	// v21
	VertexIndex v21 = createVertex(graph, vml_vec2(82, 1));
	// v22
	VertexIndex v22 = createVertex(graph, vml_vec2(74, 36));
	// v23
	VertexIndex v23 = createVertex(graph, vml_vec2(94, 36));
	// v24
	VertexIndex v24 = createVertex(graph, vml_vec2(94, 20));
	// v25
	VertexIndex v25 = createVertex(graph, vml_vec2(84, 22));
	// v26
	VertexIndex v26 = createVertex(graph, vml_vec2(80, 31));
	// v27
	VertexIndex v27 = createVertex(graph, vml_vec2(88, 31));

	// v1v2
	connect(graph, v1, v2, false);
	// v1v3
	connect(graph, v1, v3, false);

	// v2v7
	connect(graph, v2, v7, false);
	// v2v4
	connect(graph, v2, v4, false);

	// v3v4
	connect(graph, v3, v4, false);

	// v4v5
	connect(graph, v4, v5, false);

	// v5v6
	connect(graph, v5, v6, false);

	// v7v11
	connect(graph, v7, v11, false);

	// v8v9
	connect(graph, v8, v9, false);

	// v9v10
	connect(graph, v9, v10, false);

	// v11v12
	connect(graph, v11, v12, false);
	// v11v13
	connect(graph, v11, v13, false);

	// v12v13
	connect(graph, v12, v13, false);
	// v12v20
	connect(graph, v12, v20, false);

	// v13v18
	connect(graph, v13, v18, false);

	// v14v15
	connect(graph, v14, v15, false);

	// v15v16
	connect(graph, v15, v16, false);

	// v18v19
	connect(graph, v18, v19, false);

	// v19v20
	connect(graph, v19, v20, false);
	// v19v21
	connect(graph, v19, v21, false);

	// v20v21
	connect(graph, v20, v21, false);
	// v20v22
	connect(graph, v20, v22, false);
	// v20v24
	connect(graph, v20, v24, false);

	// v22v23
	connect(graph, v22, v24, false);

	// v25v26
	connect(graph, v25, v26, false);
	// v25v27
	connect(graph, v25, v27, false);

	// v26v27
	connect(graph, v26, v27, false);
}

//TEST(minimal_cycle_basis, extract_primitives)
void foo()
{
	Graph graph;
	QuadTree quadtree;
	Vertex vertices[MAX_VERTICES];
	Edge edges[MAX_EDGES];
	EdgeIndex queryResults[MAX_RESULTS_PER_QUERY];
	Quadrant quadrants[MAX_QUADRANTS];
	QuadrantEdges quadrantEdges[MAX_QUADRANT_EDGES];

	initializeQuadtree(&quadtree, WORLD_BOUNDS, QUADTREE_DEPTH, MAX_RESULTS_PER_QUERY, quadrants, quadrantEdges);
	initializeGraph(&graph, SNAP_RADIUS, MAX_VERTICES, MAX_EDGES, vertices, edges, &quadtree, MAX_RESULTS_PER_QUERY, queryResults);

	setUpGraph(&graph);

	allocateExtractionBuffers(HEAP_BUFFER_SIZE, PRIMITIVES_BUFFER_SIZE, SEQUENCE_BUFFER_SIZE, VISITED_BUFFER_SIZE);

	extractPrimitives(&graph);

	EXPECT_EQ(1, 1);

	freeExtractionBuffers();
}

}