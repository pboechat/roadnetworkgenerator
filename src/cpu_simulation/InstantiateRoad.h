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
	virtual void execute(WorkQueuesManager<Procedure>& workQueuesManager, RoadNetworkGraph::Graph& roadNetworkGraph, const Configuration& configuration);

	// TODO: improve design
	static void initialize(const Configuration& configuration);
	static void dispose();

private:
	static unsigned char* populationDensities;
	static unsigned int* distances;

	Road road;

	void evaluateGlobalGoals(const Configuration& configuration, RoadNetworkGraph::VertexIndex newOrigin, const glm::vec3& position, float length, int* delays, RoadAttributes* roadAttributes, RuleAttributes* ruleAttributes);
	void followHighestPopulationDensity(const Configuration& configuration, const glm::vec3& start, RoadAttributes& highwayRoadAttributes, RuleAttributes& highwayRuleAttributes) const;
	void applyAngleDeviation(const Configuration& configuration, RoadAttributes& roadAttributes) const;

};

#endif