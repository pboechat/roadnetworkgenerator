#ifndef DEFINES_H
#define DEFINES_H

#define MAX_VERTICES 4000
#define MAX_VERTEX_CONNECTIONS 10
// MAX_VERTICES * MAX_VERTEX_CONNECTIONS
#define MAX_EDGES 40000
#define MAX_EDGE_REFERENCIES_PER_QUERY 1000
#define MAX_DISTANCE 10000

namespace RoadNetworkGraph
{

	typedef int VertexIndex;
	typedef int EdgeIndex;

}

#endif