#ifndef DEFINES_H
#define DEFINES_H

#define MAX_VERTICES 10000
#define MAX_VERTEX_CONNECTIONS 200
// MAX_VERTICES * MAX_VERTEX_CONNECTIONS
#define MAX_EDGES 2000000
#define MAX_RESULTS_PER_QUERY 50000
#define MAX_DISTANCE 10000
#define MAX_EDGES_PER_QUADRANT 2000

namespace RoadNetworkGraph
{

typedef int VertexIndex;
typedef int EdgeIndex;
typedef int QuadrantIndex;
typedef int QuadrantEdgesIndex;

}

#endif