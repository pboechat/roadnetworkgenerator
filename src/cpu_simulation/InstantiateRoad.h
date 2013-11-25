#ifndef INSTANTIATEROAD_H
#define INSTANTIATEROAD_H

#include <Procedure.h>
#include <Road.h>

#include <glm/glm.hpp>

#define INSTANTIATE_ROAD_CODE 0

struct InstantiateRoad : public Procedure
{
	InstantiateRoad();
	InstantiateRoad(const Road& road);

	virtual unsigned int getCode() const;
	virtual void execute(WorkQueuesManager& manager, RoadNetworkGraph::Graph& graph, const Configuration& configuration);
	InstantiateRoad& operator = (const InstantiateRoad& other);

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