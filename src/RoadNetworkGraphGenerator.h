#ifndef ROADNETWORKGRAPHGENERATOR_H
#define ROADNETWORKGRAPHGENERATOR_H

#pragma once

#include <Graph.h>
#include <Vertex.h>
#include <Edge.h>
#include <Configuration.h>
#include <ImageMap.h>
#include <RoadNetworkGraphGenerationObserver.h>
#include <MathExtras.h>

#include <vector>

class RoadNetworkGraphGenerator
{
public:
	RoadNetworkGraphGenerator(const Configuration& configuration,
						 const ImageMap& populationDensityMap,
						 const ImageMap& waterBodiesMap,
						 const ImageMap& blockadesMap,
						 const ImageMap& naturalPatternMap,
						 const ImageMap& radialPatternMap,
						 const ImageMap& rasterPatternMap) : 
		configuration(configuration), 
		populationDensityMap(populationDensityMap),
		waterBodiesMap(waterBodiesMap),
		blockadesMap(blockadesMap),
		naturalPatternMap(naturalPatternMap),
		radialPatternMap(radialPatternMap),
		rasterPatternMap(rasterPatternMap),
		maxPrimitiveSize(0)
	{
	}
	~RoadNetworkGraphGenerator() {}

	inline void addObserver(RoadNetworkGraphGenerationObserver* observer)
	{
		observers.push_back(observer);
	}

	void execute();

private:
	const Configuration& configuration;
	const ImageMap& populationDensityMap;
	const ImageMap& waterBodiesMap;
	const ImageMap& blockadesMap;
	const ImageMap& naturalPatternMap;
	const ImageMap& radialPatternMap;
	const ImageMap& rasterPatternMap;
	unsigned int maxPrimitiveSize;
	std::vector<RoadNetworkGraphGenerationObserver*> observers;

	void copyGraphToDevice(Graph* graph);
	void copyGraphToHost(Graph* graph);
	void expand(unsigned int numDerivations, unsigned int startingQueue, unsigned int numQueues);
	void computeCollisions();
	void notifyObservers(Graph* graph, unsigned int numPrimitives, Primitive* primitives);

};

#endif