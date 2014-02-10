#ifndef CONSTANTS_H
#define CONSTANTS_H

#pragma once

#include <VectorMath.h>

//////////////////////////////////////////////////////////////////////////
// CAMERA
//////////////////////////////////////////////////////////////////////////

#define ZNEAR 10.0f
#define ZFAR 10000.0f
#define FOVY_DEG 60.0f

//////////////////////////////////////////////////////////////////////////
// COLORS
//////////////////////////////////////////////////////////////////////////

#define VERTEX_LABEL_COLOR vml_vec4(0.0f, 1.0f, 0.0f, 1.0f)
#define EDGE_LABEL_COLOR vml_vec4(0.0f, 0.0f, 1.0f, 1.0f)
#define WHITE_COLOR vml_vec4(1.0f, 1.0f, 1.0f, 1.0f)
#define BLACK_COLOR vml_vec4(0.0f, 0.0f, 0.0f, 1.0f)
#define WATER_COLOR vml_vec4(0.5f, 0.74f, 0.98f, 1.0f)
#define GRASS_COLOR vml_vec4(0.659f, 0.8f, 0.588f, 1.0f)

//////////////////////////////////////////////////////////////////////////
// PROCEDURES
//////////////////////////////////////////////////////////////////////////

#define NUM_PROCEDURES 6

//////////////////////////////////////////////////////////////////////////
//	WORK QUEUES
//////////////////////////////////////////////////////////////////////////
#define MAX_NUM_WORKITEMS 5000
#define WORK_ITEM_SIZE 100
// MAX_NUM_WORKITEMS * WORK_ITEM_SIZE
#define WORK_QUEUE_DATA_SIZE 500000

//////////////////////////////////////////////////////////////////////////
// CONFIGURATION
//////////////////////////////////////////////////////////////////////////

#define MAX_SPAWN_POINTS 100
#define MAX_CONFIGURATION_STRING_SIZE 128

//////////////////////////////////////////////////////////////////////////
// PARSING PATTERNS
//////////////////////////////////////////////////////////////////////////

#define VEC2_VECTOR_PATTERN "(\\([^\\)]+\\)\\,?)"

//////////////////////////////////////////////////////////////////////////
// GRAPH
//////////////////////////////////////////////////////////////////////////

#define MAX_VERTEX_IN_CONNECTIONS 40
#define MAX_VERTEX_OUT_CONNECTIONS 40
// MAX_VERTEX_ADJACENCIES = MAX_VERTEX_IN_CONNECTIONS + MAX_VERTEX_OUT_CONNECTIONS
#define MAX_VERTEX_ADJACENCIES 80
#define MAX_EDGES_PER_QUADRANT 1000
#define MAX_EDGES_PER_PRIMITIVE 100
#define MAX_VERTICES_PER_PRIMITIVE 200
#define MAX_RESULTS_PER_QUERY 10
#define QUADTREE_STACK_DATA_SIZE 91

//////////////////////////////////////////////////////////////////////////
//	GENERAL
//////////////////////////////////////////////////////////////////////////

#define FONT_FILE_PATH "../../../../data/fonts/arial.glf"

#ifdef COLLECT_STATISTICS
//////////////////////////////////////////////////////////////////////////
// DEBUG MACROS
//////////////////////////////////////////////////////////////////////////

#define toKilobytes(a) (a / 1024)
#define toMegabytes(a) (a / 1048576)
#endif

#endif