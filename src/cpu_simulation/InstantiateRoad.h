#ifndef INSTANTIATEROAD_H
#define INSTANTIATEROAD_H

#include <Procedure.h>
#include <Road.h>
#include <Pattern.h>

#include <glm/glm.hpp>

#define INSTANTIATE_ROAD_CODE 0

struct InstantiateRoad : public Procedure
{
	InstantiateRoad();
	InstantiateRoad(const Road& road);

	virtual unsigned int getCode() const;
	virtual void execute(WorkQueuesManager& manager, RoadNetworkGraph::Graph& graph, const Configuration& configuration);

	// TODO: improve design
	static void initialize(const Configuration& configuration);
	static void dispose();

private:
	static unsigned char* populationDensities;
	static unsigned int* distances;

	Road road;

	void evaluateGlobalGoals(const Configuration& configuration, RoadNetworkGraph::VertexIndex newOrigin, const glm::vec3& position, int* delays, RoadAttributes* roadAttributes, RuleAttributes* ruleAttributes);
	void findHighestPopulationDensity(const Configuration& configuration, const glm::vec3& start, float startingAngle, glm::vec3& goal, unsigned int& distance) const;
	Pattern findUnderlyingPattern(const Configuration& configuration, const glm::vec3& position) const;
	void applyHighwayGoalDeviation(const Configuration& configuration, RoadAttributes& roadAttributes) const;
	void applyNaturalPatternRule(const Configuration& configuration, const glm::vec3& position, unsigned int goalDistance, int& delay, RoadAttributes& roadAttributes, RuleAttributes& ruleAttributes) const;
	void applyRadialPatternRule(const Configuration& configuration, const glm::vec3& position, unsigned int goalDistance, int& delay, RoadAttributes& roadAttributes, RuleAttributes& ruleAttributes) const;
	void applyRasterPatternRule(const Configuration& configuration, const glm::vec3& position, unsigned int goalDistance, int& delay, RoadAttributes& roadAttributes, RuleAttributes& ruleAttributes) const;

};

#endif