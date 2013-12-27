#ifndef ROADNETWORKGENERATOR_H
#define ROADNETWORKGENERATOR_H

#include <RoadNetworkGeometry.h>
#include <Globals.h>
#include <Procedures.h>
#include <Road.h>
#include <RoadAttributes.h>
#include <RuleAttributes.h>
#include <MathExtras.h>

#include <vector_math.h>

class RoadNetworkGenerator
{
public:
	RoadNetworkGenerator(unsigned int maxWorkQueueCapacity) : maxWorkQueueCapacity(maxWorkQueueCapacity), buffer1(g_workQueues1, NUM_PROCEDURES), buffer2(g_workQueues2, NUM_PROCEDURES), lastDerivation(0)
#ifdef _DEBUG
		, maxWorkQueueCapacityUsed(0)
#endif
	{}
	~RoadNetworkGenerator() {}

	void execute()
	{
		WorkQueues* frontBuffer = &buffer1;
		WorkQueues* backBuffer = &buffer2;

		for (unsigned int i = 0; i < g_configuration->numSpawnPoints; i++)
		{
			vml_vec2 spawnPoint = g_configuration->spawnPoints[i];
			RoadNetworkGraph::VertexIndex source = RoadNetworkGraph::createVertex(g_graph, spawnPoint);
			frontBuffer->addWorkItem(EVALUATE_ROAD, Road(0, RoadAttributes(source, g_configuration->highwayLength, 0, true), RuleAttributes(), UNASSIGNED));
			frontBuffer->addWorkItem(EVALUATE_ROAD, Road(0, RoadAttributes(source, g_configuration->highwayLength, -MathExtras::HALF_PI, true), RuleAttributes(), UNASSIGNED));
			frontBuffer->addWorkItem(EVALUATE_ROAD, Road(0, RoadAttributes(source, g_configuration->highwayLength, MathExtras::HALF_PI, true), RuleAttributes(), UNASSIGNED));
			frontBuffer->addWorkItem(EVALUATE_ROAD, Road(0, RoadAttributes(source, g_configuration->highwayLength, MathExtras::PI, true), RuleAttributes(), UNASSIGNED));
		}

		lastDerivation = 0;

		while (frontBuffer->notEmpty() && lastDerivation++ < g_configuration->maxDerivations)
		{
#ifdef _DEBUG
			if (frontBuffer->getNumWorkItems() > maxWorkQueueCapacityUsed)
			{
				maxWorkQueueCapacityUsed = frontBuffer->getNumWorkItems();
			}
#endif
			frontBuffer->executeAllWorkItems(backBuffer);
			std::swap(frontBuffer, backBuffer);
		}

		if (g_configuration->removeDeadEndRoads)
		{
			RoadNetworkGraph::removeDeadEndRoads(g_graph);
		}
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
	WorkQueues buffer1;
	WorkQueues buffer2;
	unsigned int maxWorkQueueCapacity;
	unsigned int lastDerivation;
#ifdef _DEBUG
	unsigned int maxWorkQueueCapacityUsed;
#endif

};

#endif