#ifndef ROADNETWORKGENERATOR_H
#define ROADNETWORKGENERATOR_H

#include <Configuration.h>
#include <RoadNetworkGeometry.h>
#include <Procedure.h>
#include <WorkQueuesManager.h>
#include <EvaluateRoad.h>
#include <RoadAttributes.h>
#include <RuleAttributes.h>
#include <Road.h>
#include <RoadNetwork.h>

#include <glm/glm.hpp>

class RoadNetworkGenerator
{
public:
	RoadNetworkGenerator() {}
	~RoadNetworkGenerator() {}

	void execute(const Configuration& configuration, RoadNetwork::Graph& roadNetwork)
	{
		WorkQueuesManager<Procedure>* frontBuffer = &buffer1;
		WorkQueuesManager<Procedure>* backBuffer = &buffer2;

		RoadAttributes initialRoadAttributes(0, configuration.highwayLength, 0, true);
		RuleAttributes initialRuleAttributes;
		initialRuleAttributes.highwayBranchingDistance = configuration.minHighwayBranchingDistance;
		frontBuffer->addWorkItem(new EvaluateRoad(Road(0, initialRoadAttributes, initialRuleAttributes, UNASSIGNED)));

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

			std::swap(frontBuffer, backBuffer);
		}

		frontBuffer->clear();
		backBuffer->clear();
	}

private:
	WorkQueuesManager<Procedure> buffer1;
	WorkQueuesManager<Procedure> buffer2;

};

#endif