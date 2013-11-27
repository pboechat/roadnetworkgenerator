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

#include <glm/glm.hpp>

#define NUM_WORK_QUEUES 3
#define WORK_QUEUE_CAPACITY 100000

class RoadNetworkGenerator
{
public:
	RoadNetworkGenerator() : buffer1(NUM_WORK_QUEUES, WORK_QUEUE_CAPACITY), buffer2(NUM_WORK_QUEUES, WORK_QUEUE_CAPACITY) {}
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
			frontBuffer->addWorkItem(EvaluateRoad(Road(0, RoadAttributes(source, configuration.highwayLength, -90, true), RuleAttributes(), UNASSIGNED)));
			frontBuffer->addWorkItem(EvaluateRoad(Road(0, RoadAttributes(source, configuration.highwayLength, 90, true), RuleAttributes(), UNASSIGNED)));
			frontBuffer->addWorkItem(EvaluateRoad(Road(0, RoadAttributes(source, configuration.highwayLength, 180, true), RuleAttributes(), UNASSIGNED)));
		}

		// TODO: improve design
		InstantiateRoad::initialize(configuration);

		unsigned int derivation = 0;
		while (frontBuffer->notEmpty() && derivation++ < configuration.maxDerivations)
		{
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

private:
	WorkQueuesManager buffer1;
	WorkQueuesManager buffer2;

};

#endif