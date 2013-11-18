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

#include <glm/glm.hpp>

#include <vector>

class RoadNetworkGenerator
{
public:
	RoadNetworkGenerator() {}
	~RoadNetworkGenerator() {}

	void execute(Configuration& configuration, RoadNetworkGeometry& geometry)
	{
		segments.clear();

		WorkQueuesManager<Procedure>* frontBuffer = &buffer1;
		WorkQueuesManager<Procedure>* backBuffer = &buffer2;
		RoadAttributes initialRoadAttributes(glm::vec3(configuration.worldWidth / 2.0f, configuration.worldWidth / 2.0f, 0), configuration.roadLength, 0, true);
		RuleAttributes initialRuleAttributes;
		frontBuffer->addWorkItem(new EvaluateRoad(Road(0, initialRoadAttributes, initialRuleAttributes, UNASSIGNED)));
		int derivation = 0;

		while (frontBuffer->notEmpty() && derivation++ < configuration.maxDerivations)
		{
			frontBuffer->resetCursors();

			do
			{
				Procedure* procedure;

				while ((procedure = frontBuffer->popWorkItem()) != 0)
				{
					procedure->execute(*backBuffer, segments, configuration.populationDensityMap, configuration.waterBodiesMap);
					delete procedure;
				}
			}
			while (frontBuffer->nextWorkQueue());

			std::swap(frontBuffer, backBuffer);
		}

		frontBuffer->clear();
		backBuffer->clear();

		geometry.build(segments);
	}

private:
	WorkQueuesManager<Procedure> buffer1;
	WorkQueuesManager<Procedure> buffer2;
	std::vector<Segment> segments;

};

#endif