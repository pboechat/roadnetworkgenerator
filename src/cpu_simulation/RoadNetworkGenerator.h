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

class RoadNetworkGenerator
{
public:
	RoadNetworkGenerator() {}
	~RoadNetworkGenerator() {}

	void execute(const Configuration& configuration, RoadNetworkGraph::Graph& roadNetwork)
	{
		WorkQueuesManager<Procedure>* frontBuffer = &buffer1;
		WorkQueuesManager<Procedure>* backBuffer = &buffer2;

		frontBuffer->addWorkItem(new EvaluateRoad(Road(0, RoadAttributes(0, configuration.highwayLength, 0, true), RuleAttributes(), UNASSIGNED)));
		frontBuffer->addWorkItem(new EvaluateRoad(Road(0, RoadAttributes(0, configuration.highwayLength, -90, true), RuleAttributes(), UNASSIGNED)));
		frontBuffer->addWorkItem(new EvaluateRoad(Road(0, RoadAttributes(0, configuration.highwayLength, 90, true), RuleAttributes(), UNASSIGNED)));
		frontBuffer->addWorkItem(new EvaluateRoad(Road(0, RoadAttributes(0, configuration.highwayLength, 180, true), RuleAttributes(), UNASSIGNED)));

		// TODO: improve design
		InstantiateRoad::initialize(configuration);

		unsigned int derivation = 0;
		while (frontBuffer->notEmpty() && derivation++ < configuration.maxDerivations)
		{
			frontBuffer->resetCursors();

			do
			{
				Procedure* procedure;

				while ((procedure = frontBuffer->popWorkItem()) != 0)
				{
					procedure->execute(*backBuffer, roadNetwork, configuration);
					delete procedure;
				}
			}
			while (frontBuffer->nextWorkQueue());

			// FIXME: checking invariants
			if (frontBuffer->notEmpty())
			{
				throw std::exception("frontBuffer->notEmpty()");
			}

			std::swap(frontBuffer, backBuffer);
		}

		if (configuration.removeDeadEndRoads)
		{
			roadNetwork.removeDeadEndRoads();
		}

		// TODO: improve design
		InstantiateRoad::dispose();

		frontBuffer->clear();
		backBuffer->clear();
	}

private:
	WorkQueuesManager<Procedure> buffer1;
	WorkQueuesManager<Procedure> buffer2;

};

#endif