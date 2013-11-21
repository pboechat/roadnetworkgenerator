#ifndef INSTANTIATEROAD_H
#define INSTANTIATEROAD_H

#include <Procedure.h>
#include <Road.h>

#include <glm/glm.hpp>

class InstantiateRoad : public Procedure
{
public:
	InstantiateRoad(const Road& road);

	virtual unsigned int getCode();
	virtual void execute(WorkQueuesManager<Procedure>& workQueuesManager, RoadNetwork::Graph& roadNetworkGraph, const Configuration& configuration);

private:
	Road road;

	void evaluateGlobalGoals(const Configuration& configuration, RoadNetwork::VertexIndex newOrigin, const glm::vec3& position, int* delays, RoadAttributes* roadAttributes, RuleAttributes* ruleAttributes);
	void followHighestPopulationDensity(const Configuration& configuration, const glm::vec3& start, RoadAttributes& highwayRoadAttributes, RuleAttributes& highwayRuleAttributes) const;

};

#endif