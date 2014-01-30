#ifndef ROADNETWORKGENERATIONOBSERVER_H
#define ROADNETWORKGENERATIONOBSERVER_H

#pragma once

#include <Graph.h>
#include <Primitive.h>

class RoadNetworkGraphGenerationObserver
{
public:
	virtual void update(Graph* graph, unsigned int numPrimitives, Primitive* primitives) = 0;

};

#endif