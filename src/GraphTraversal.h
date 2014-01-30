#ifndef GRAPHTRAVERSAL_H
#define GRAPHTRAVERSAL_H

#pragma once

#include <Vertex.h>
#include <Edge.h>

struct GraphTraversal
{
	virtual bool operator () (const Vertex& source, const Vertex& destination, const Edge& edge) = 0;

};


#endif