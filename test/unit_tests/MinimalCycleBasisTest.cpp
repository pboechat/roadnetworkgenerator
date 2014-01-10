#include <Graph.h>
#include <QuadTree.h>
#include <MinimalCycleBasis.h>
#include <Box2D.h>
#include <Array.h>
#include <vector_math.h>

#include <gtest/gtest.h>

namespace RoadNetworkGraph
{

#define WORLD_BOUNDS Box2D(vml_vec2(0, 0), vml_vec2(400, 200))
#define SNAP_RADIUS 0.0f
#define MAX_VERTICES 28
#define MAX_EDGES 29
#define MAX_RESULTS_PER_QUERY 1
#define HEAP_BUFFER_SIZE 28
#define PRIMITIVES_BUFFER_SIZE 12
#define SEQUENCE_BUFFER_SIZE 29
#define VISITED_BUFFER_SIZE 28
#define QUADTREE_DEPTH 1
#define MAX_QUADRANTS 1
#define MAX_QUADRANT_EDGES 29

#define V0_POS vml_vec2(11, 188)
#define V1_POS vml_vec2(22, 168)
#define V2_POS vml_vec2(74, 164)
#define V3_POS vml_vec2(3, 113)
#define V4_POS vml_vec2(90, 113)
#define V5_POS vml_vec2(51, 138)
#define V6_POS vml_vec2(22, 124)
#define V7_POS vml_vec2(118, 181)
#define V8_POS vml_vec2(113, 134)
#define V9_POS vml_vec2(95, 90)
#define V10_POS vml_vec2(129, 91)
#define V11_POS vml_vec2(133, 175)
#define V12_POS vml_vec2(140, 117)
#define V13_POS vml_vec2(211, 179)
#define V14_POS vml_vec2(180, 66)
#define V15_POS vml_vec2(213, 99)
#define V16_POS vml_vec2(261, 71)
#define V17_POS vml_vec2(210, 140)
#define V18_POS vml_vec2(249, 156)
#define V19_POS vml_vec2(282, 177)
#define V20_POS vml_vec2(292, 125)
#define V21_POS vml_vec2(324, 197)
#define V22_POS vml_vec2(292, 61)
#define V23_POS vml_vec2(374, 61)
#define V24_POS vml_vec2(374, 125)
#define V25_POS vml_vec2(332, 116)
#define V26_POS vml_vec2(318, 78)
#define V27_POS vml_vec2(348, 78)

void setUpGraph(Graph* graph)
{
	// v0
	VertexIndex v0 = createVertex(graph, V0_POS);
	// v1
	VertexIndex v1 = createVertex(graph, V1_POS);
	// v2
	VertexIndex v2 = createVertex(graph, V2_POS);
	// v3
	VertexIndex v3 = createVertex(graph, V3_POS);
	// v4
	VertexIndex v4 = createVertex(graph, V4_POS);
	// v5
	VertexIndex v5 = createVertex(graph, V5_POS);
	// v6
	VertexIndex v6 = createVertex(graph, V6_POS);
	// v7
	VertexIndex v7 = createVertex(graph, V7_POS);
	// v8
	VertexIndex v8 = createVertex(graph, V8_POS);
	// v9
	VertexIndex v9 = createVertex(graph, V9_POS);
	// v10
	VertexIndex v10 = createVertex(graph, V10_POS);
	// v11
	VertexIndex v11 = createVertex(graph, V11_POS);
	// v12
	VertexIndex v12 = createVertex(graph, V12_POS);
	// v13
	VertexIndex v13 = createVertex(graph, V13_POS);
	// v14
	VertexIndex v14 = createVertex(graph, V14_POS);
	// v15
	VertexIndex v15 = createVertex(graph, V15_POS);
	// v16
	VertexIndex v16 = createVertex(graph, V16_POS);
	// v17
	VertexIndex v17 = createVertex(graph, V17_POS);
	// v18
	VertexIndex v18 = createVertex(graph, V18_POS);
	// v19
	VertexIndex v19 = createVertex(graph, V19_POS);
	// v20
	VertexIndex v20 = createVertex(graph, V20_POS);
	// v21
	VertexIndex v21 = createVertex(graph, V21_POS);
	// v22
	VertexIndex v22 = createVertex(graph, V22_POS);
	// v23
	VertexIndex v23 = createVertex(graph, V23_POS);
	// v24
	VertexIndex v24 = createVertex(graph, V24_POS);
	// v25
	VertexIndex v25 = createVertex(graph, V25_POS);
	// v26
	VertexIndex v26 = createVertex(graph, V26_POS);
	// v27
	VertexIndex v27 = createVertex(graph, V27_POS);
	// v1v2
	connect(graph, v1, v2, false);
	// v1v3
	connect(graph, v1, v3, false);
	// v2v4
	connect(graph, v2, v4, false);
	// v2v7
	connect(graph, v2, v7, false);
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
	// v8v10
	connect(graph, v8, v10, false);
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
	connect(graph, v22, v23, false);
	// v23v24
	connect(graph, v23, v24, false);
	// v25v26
	connect(graph, v25, v26, false);
	// v25v27
	connect(graph, v25, v27, false);
	// v26v27
	connect(graph, v26, v27, false);
}

TEST(minimal_cycle_basis, extract_primitives)
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
	Array<Primitive>& primitives = extractPrimitives(&graph);
	EXPECT_EQ(12, primitives.size());
	// v6v5v4 (filament)
	EXPECT_EQ(3, primitives[0].numVertices);
	EXPECT_EQ(V6_POS, primitives[0].vertices[0]);
	EXPECT_EQ(V5_POS, primitives[0].vertices[1]);
	EXPECT_EQ(V4_POS, primitives[0].vertices[2]);
	// v3v4v2v1 (minimal cycle basis)
	EXPECT_EQ(4, primitives[1].numVertices);
	EXPECT_EQ(V3_POS, primitives[1].vertices[0]);
	EXPECT_EQ(V4_POS, primitives[1].vertices[1]);
	EXPECT_EQ(V2_POS, primitives[1].vertices[2]);
	EXPECT_EQ(V1_POS, primitives[1].vertices[3]);
	// v0 (isolated vertex)
	EXPECT_EQ(1, primitives[2].numVertices);
	EXPECT_EQ(V0_POS, primitives[2].vertices[0]);
	// v2v7v11 (filament)
	EXPECT_EQ(3, primitives[3].numVertices);
	EXPECT_EQ(V2_POS, primitives[3].vertices[0]);
	EXPECT_EQ(V7_POS, primitives[3].vertices[1]);
	EXPECT_EQ(V11_POS, primitives[3].vertices[2]);
	// v9v10v8 (minimal cycle basis)
	EXPECT_EQ(3, primitives[4].numVertices);
	EXPECT_EQ(V9_POS, primitives[4].vertices[0]);
	EXPECT_EQ(V10_POS, primitives[4].vertices[1]);
	EXPECT_EQ(V8_POS, primitives[4].vertices[2]);
	// v11v13v12 (minimal cycle basis)
	EXPECT_EQ(3, primitives[5].numVertices);
	EXPECT_EQ(V11_POS, primitives[5].vertices[0]);
	EXPECT_EQ(V13_POS, primitives[5].vertices[1]);
	EXPECT_EQ(V12_POS, primitives[5].vertices[2]);
	// v12v20v19v18v13 (minimal cycle basis)
	EXPECT_EQ(5, primitives[6].numVertices);
	EXPECT_EQ(V12_POS, primitives[6].vertices[0]);
	EXPECT_EQ(V20_POS, primitives[6].vertices[1]);
	EXPECT_EQ(V19_POS, primitives[6].vertices[2]);
	EXPECT_EQ(V18_POS, primitives[6].vertices[3]);
	EXPECT_EQ(V13_POS, primitives[6].vertices[4]);
	// v14v15v16 (filament)
	EXPECT_EQ(3, primitives[7].numVertices);
	EXPECT_EQ(V14_POS, primitives[7].vertices[0]);
	EXPECT_EQ(V15_POS, primitives[7].vertices[1]);
	EXPECT_EQ(V16_POS, primitives[7].vertices[2]);
	// v17 (isolated vertex)
	EXPECT_EQ(1, primitives[8].numVertices);
	EXPECT_EQ(V17_POS, primitives[8].vertices[0]);
	// v19v21v20 (minimal cycle basis)
	EXPECT_EQ(3, primitives[9].numVertices);
	EXPECT_EQ(V19_POS, primitives[9].vertices[0]);
	EXPECT_EQ(V21_POS, primitives[9].vertices[1]);
	EXPECT_EQ(V20_POS, primitives[9].vertices[2]);
	// v22v24v23v20 (minimal cycle basis)
	EXPECT_EQ(4, primitives[10].numVertices);
	EXPECT_EQ(V22_POS, primitives[10].vertices[0]);
	EXPECT_EQ(V24_POS, primitives[10].vertices[1]);
	EXPECT_EQ(V23_POS, primitives[10].vertices[2]);
	EXPECT_EQ(V20_POS, primitives[10].vertices[3]);
	// v26v27v25 (minimal cycle basis)
	EXPECT_EQ(3, primitives[11].numVertices);
	EXPECT_EQ(V26_POS, primitives[11].vertices[0]);
	EXPECT_EQ(V27_POS, primitives[11].vertices[1]);
	EXPECT_EQ(V25_POS, primitives[11].vertices[2]);
	freeExtractionBuffers();
}

}