#include <Graph.h>
#include <GraphFunctions.cuh>
#include <QuadTree.h>
#include <MinimalCycleBasis.h>
#include <Box2D.h>
#include <Array.h>
#include <VectorMath.h>

#include <gtest/gtest.h>

namespace RoadNetworkGraph
{

#define WORLD_BOUNDS Box2D(vml_vec2(0, 0), vml_vec2(400, 200))
#define SNAP_RADIUS 0.0f
#define MAX_VERTICES 28
#define MAX_EDGES 30
#define MAX_PRIMITIVES 12
#define HEAP_BUFFER_SIZE 28
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
	int v0 = createVertex(graph, V0_POS);
	// v1
	int v1 = createVertex(graph, V1_POS);
	// v2
	int v2 = createVertex(graph, V2_POS);
	// v3
	int v3 = createVertex(graph, V3_POS);
	// v4
	int v4 = createVertex(graph, V4_POS);
	// v5
	int v5 = createVertex(graph, V5_POS);
	// v6
	int v6 = createVertex(graph, V6_POS);
	// v7
	int v7 = createVertex(graph, V7_POS);
	// v8
	int v8 = createVertex(graph, V8_POS);
	// v9
	int v9 = createVertex(graph, V9_POS);
	// v10
	int v10 = createVertex(graph, V10_POS);
	// v11
	int v11 = createVertex(graph, V11_POS);
	// v12
	int v12 = createVertex(graph, V12_POS);
	// v13
	int v13 = createVertex(graph, V13_POS);
	// v14
	int v14 = createVertex(graph, V14_POS);
	// v15
	int v15 = createVertex(graph, V15_POS);
	// v16
	int v16 = createVertex(graph, V16_POS);
	// v17
	int v17 = createVertex(graph, V17_POS);
	// v18
	int v18 = createVertex(graph, V18_POS);
	// v19
	int v19 = createVertex(graph, V19_POS);
	// v20
	int v20 = createVertex(graph, V20_POS);
	// v21
	int v21 = createVertex(graph, V21_POS);
	// v22
	int v22 = createVertex(graph, V22_POS);
	// v23
	int v23 = createVertex(graph, V23_POS);
	// v24
	int v24 = createVertex(graph, V24_POS);
	// v25
	int v25 = createVertex(graph, V25_POS);
	// v26
	int v26 = createVertex(graph, V26_POS);
	// v27
	int v27 = createVertex(graph, V27_POS);
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
	Quadrant quadrants[MAX_QUADRANTS];
	QuadrantEdges quadrantEdges[MAX_QUADRANT_EDGES];
	initializeQuadtreeOnHost(&quadtree, WORLD_BOUNDS, QUADTREE_DEPTH, MAX_QUADRANTS, quadrants, quadrantEdges);
	initializeGraphOnHost(&graph, SNAP_RADIUS, MAX_VERTICES, MAX_EDGES, vertices, edges, &quadtree);
	setUpGraph(&graph);
	Primitive primitives[MAX_PRIMITIVES];
	unsigned int numExtractedPrimitives = extractPrimitives(&graph, primitives, MAX_PRIMITIVES);
	EXPECT_EQ(12, numExtractedPrimitives);
	// v6v5v4 (filament)
	EXPECT_EQ(3, primitives[0].numVertices);
	EXPECT_EQ(6, primitives[0].vertices[0]);
	EXPECT_EQ(5, primitives[0].vertices[1]);
	EXPECT_EQ(4, primitives[0].vertices[2]);
	// v3v4v2v1 (minimal cycle basis)
	EXPECT_EQ(4, primitives[1].numVertices);
	EXPECT_EQ(3, primitives[1].vertices[0]);
	EXPECT_EQ(4, primitives[1].vertices[1]);
	EXPECT_EQ(2, primitives[1].vertices[2]);
	EXPECT_EQ(1, primitives[1].vertices[3]);
	// v0 (isolated vertex)
	EXPECT_EQ(1, primitives[2].numVertices);
	EXPECT_EQ(0, primitives[2].vertices[0]);
	// v2v7v11 (filament)
	EXPECT_EQ(3, primitives[3].numVertices);
	EXPECT_EQ(2, primitives[3].vertices[0]);
	EXPECT_EQ(7, primitives[3].vertices[1]);
	EXPECT_EQ(11, primitives[3].vertices[2]);
	// v9v10v8 (minimal cycle basis)
	EXPECT_EQ(3, primitives[4].numVertices);
	EXPECT_EQ(9, primitives[4].vertices[0]);
	EXPECT_EQ(10, primitives[4].vertices[1]);
	EXPECT_EQ(8, primitives[4].vertices[2]);
	// v11v12v13 (minimal cycle basis)
	EXPECT_EQ(3, primitives[5].numVertices);
	EXPECT_EQ(11, primitives[5].vertices[0]);
	EXPECT_EQ(12, primitives[5].vertices[1]);
	EXPECT_EQ(13, primitives[5].vertices[2]);
	// v12v20v19v18v13 (minimal cycle basis)
	EXPECT_EQ(5, primitives[6].numVertices);
	EXPECT_EQ(12, primitives[6].vertices[0]);
	EXPECT_EQ(20, primitives[6].vertices[1]);
	EXPECT_EQ(19, primitives[6].vertices[2]);
	EXPECT_EQ(18, primitives[6].vertices[3]);
	EXPECT_EQ(13, primitives[6].vertices[4]);
	// v14v15v16 (filament)
	EXPECT_EQ(3, primitives[7].numVertices);
	EXPECT_EQ(14, primitives[7].vertices[0]);
	EXPECT_EQ(15, primitives[7].vertices[1]);
	EXPECT_EQ(16, primitives[7].vertices[2]);
	// v17 (isolated vertex)
	EXPECT_EQ(1, primitives[8].numVertices);
	EXPECT_EQ(17, primitives[8].vertices[0]);
	// v19v20v21 (minimal cycle basis)
	EXPECT_EQ(3, primitives[9].numVertices);
	EXPECT_EQ(19, primitives[9].vertices[0]);
	EXPECT_EQ(20, primitives[9].vertices[1]);
	EXPECT_EQ(21, primitives[9].vertices[2]);
	// v22v23v24v20 (minimal cycle basis)
	EXPECT_EQ(4, primitives[10].numVertices);
	EXPECT_EQ(22, primitives[10].vertices[0]);
	EXPECT_EQ(23, primitives[10].vertices[1]);
	EXPECT_EQ(24, primitives[10].vertices[2]);
	EXPECT_EQ(20, primitives[10].vertices[3]);
	// v26v27v25 (minimal cycle basis)
	EXPECT_EQ(3, primitives[11].numVertices);
	EXPECT_EQ(26, primitives[11].vertices[0]);
	EXPECT_EQ(27, primitives[11].vertices[1]);
	EXPECT_EQ(25, primitives[11].vertices[2]);
}

}