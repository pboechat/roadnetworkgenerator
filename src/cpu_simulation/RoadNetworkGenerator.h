#ifndef ROADNETWORKGENERATOR_H
#define ROADNETWORKGENERATOR_H

#include <Configuration.h>
#include <RoadNetworkGeometry.h>
#include <Procedure.h>
#include <InstantiateRoad.h>
#include <WorkQueuesManager.h>
#include <EvaluateRoad.h>
#include <RoadAttributes.h>
#include <RuleAttributes.h>
#include <Road.h>
#include <Graph.h>
#include <MathExtras.h>

#include <glm/glm.hpp>

#define NUM_WORK_QUEUES 3

class RoadNetworkGenerator
{
public:
	RoadNetworkGenerator(unsigned int maxWorkQueueCapacity) : maxWorkQueueCapacity(maxWorkQueueCapacity), buffer1(NUM_WORK_QUEUES, maxWorkQueueCapacity), buffer2(NUM_WORK_QUEUES, maxWorkQueueCapacity), lastDerivation(0)
#ifdef _DEBUG
	,maxWorkQueueCapacityUsed(0) 
#endif
	{}
	~RoadNetworkGenerator() {}

	void execute(const Configuration& configuration, RoadNetworkGraph::Graph& graph)
	{
		WorkQueuesManager* frontBuffer = &buffer1;
		WorkQueuesManager* backBuffer = &buffer2;

		for (unsigned int i = 0; i < configuration.numSpawnPoints; i++)
		{
			glm::vec3 spawnPoint = configuration.spawnPoints[i];

			RoadNetworkGraph::VertexIndex source = graph.createVertex(spawnPoint);

			frontBuffer->addWorkItem(EvaluateRoad(Road(0, RoadAttributes(source, configuration.highwayLength, 0, true), RuleAttributes(), UNASSIGNED)));
			frontBuffer->addWorkItem(EvaluateRoad(Road(0, RoadAttributes(source, configuration.highwayLength, -MathExtras::HALF_PI, true), RuleAttributes(), UNASSIGNED)));
			frontBuffer->addWorkItem(EvaluateRoad(Road(0, RoadAttributes(source, configuration.highwayLength, MathExtras::HALF_PI, true), RuleAttributes(), UNASSIGNED)));
			frontBuffer->addWorkItem(EvaluateRoad(Road(0, RoadAttributes(source, configuration.highwayLength, MathExtras::PI, true), RuleAttributes(), UNASSIGNED)));
		}

		// TODO: improve design
		InstantiateRoad::initialize(configuration);

		lastDerivation = 0;
		while (frontBuffer->notEmpty() && lastDerivation++ < configuration.maxDerivations)
		{
#ifdef _DEBUG
			if (frontBuffer->size() > maxWorkQueueCapacityUsed)
			{
				maxWorkQueueCapacityUsed = frontBuffer->size();
			}
#endif

			frontBuffer->executeAllWorkItems(*backBuffer, graph, configuration);
			std::swap(frontBuffer, backBuffer);
		}

		if (configuration.removeDeadEndRoads)
		{
			graph.removeDeadEndRoads();
		}

		// TODO: improve design
		InstantiateRoad::dispose();
	}

#ifdef _DEBUG
	inline unsigned int getLastStep() const
	{
		return lastDerivation - 1;
	}

	inline unsigned int getMaxWorkQueueCapacity() const
	{
		return maxWorkQueueCapacity;
	}

	inline unsigned int getMaxWorkQueueCapacityUsed() const
	{
		return maxWorkQueueCapacityUsed;
	}
#endif

private:
	WorkQueuesManager buffer1;
	WorkQueuesManager buffer2;
	unsigned int maxWorkQueueCapacity;
	unsigned int lastDerivation;
#ifdef _DEBUG
	unsigned int maxWorkQueueCapacityUsed;
#endif

};

#endif