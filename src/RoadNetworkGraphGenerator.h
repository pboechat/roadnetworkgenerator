#ifndef ROADNETWORKGRAPHGENERATOR_H
#define ROADNETWORKGRAPHGENERATOR_H

#include "Defines.h"
#include <Graph.cuh>
#include <Configuration.cuh>
#include <ImageMap.cuh>
#include <RoadNetworkGraphGenerationObserver.h>

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
		lastHighwayDerivation(0), 
		lastStreetDerivation(0), 
		maxPrimitiveSize(0)
#ifdef _DEBUG
		, maxWorkQueueCapacityUsed(0)
#endif
	{
	}
	~RoadNetworkGraphGenerator() {}

	inline void addObserver(RoadNetworkGraphGenerationObserver* observer)
	{
		observers.push_back(observer);
	}

	void execute();

#ifdef _DEBUG
	inline unsigned int getLastHighwayDerivation() const
	{
		return lastHighwayDerivation - 1;
	}

	inline unsigned int getLastStreetDerivation() const
	{
		return lastStreetDerivation - 1;
	}

	inline unsigned int getMaxWorkQueueCapacityUsed() const
	{
		return maxWorkQueueCapacityUsed;
	}

	inline unsigned int getMaxPrimitiveSize() const
	{
		return maxPrimitiveSize;
	}
#endif

private:
	const Configuration& configuration;
	const ImageMap& populationDensityMap;
	const ImageMap& waterBodiesMap;
	const ImageMap& blockadesMap;
	const ImageMap& naturalPatternMap;
	const ImageMap& radialPatternMap;
	const ImageMap& rasterPatternMap;
	unsigned int maxWorkQueueCapacity;
	unsigned int maxPrimitiveSize;
	unsigned int lastHighwayDerivation;
	unsigned int lastStreetDerivation;
	std::vector<RoadNetworkGraphGenerationObserver*> observers;
#ifdef _DEBUG
	unsigned int maxWorkQueueCapacityUsed;
#endif

	void copyGraphToDevice(RoadNetworkGraph::Graph* graph);
	void copyGraphToHost(RoadNetworkGraph::Graph* graph);
	void expand(unsigned int numDerivations);
	void notifyObservers(RoadNetworkGraph::Graph* graph, unsigned int numPrimitives, RoadNetworkGraph::Primitive* primitives);

};

#endif