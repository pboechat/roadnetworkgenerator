#ifndef DEFINES_H
#define DEFINES_H

#define MAX_VERTICES 5000
#define MAX_VERTEX_CONNECTIONS 200
// MAX_VERTICES * MAX_VERTEX_CONNECTIONS
#define MAX_EDGES 1000000
#define MAX_RESULTS_PER_QUERY 10000
#define MAX_DISTANCE 10000
#define MAX_EDGES_PER_QUADRANT 1000

namespace RoadNetworkGraph
{

typedef int VertexIndex;
typedef int EdgeIndex;
typedef int QuadrantIndex;
typedef int QuadrantEdgesIndex;

}

#endif