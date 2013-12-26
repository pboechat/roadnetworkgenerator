#ifndef INSTANTIATEROAD_H
#define INSTANTIATEROAD_H

#include <Procedure.h>
#include <Road.h>
#include <Pattern.h>

#include <vector_math.h>

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

	void evaluateGlobalGoals(const Configuration& configuration, RoadNetworkGraph::VertexIndex newOrigin, const vml_vec2& position, int* delays, RoadAttributes* roadAttributes, RuleAttributes* ruleAttributes);
	void findHighestPopulationDensity(const Configuration& configuration, const vml_vec2& start, float startingAngle, vml_vec2& goal, unsigned int& distance) const;
	Pattern findUnderlyingPattern(const Configuration& configuration, const vml_vec2& position) const;
	void applyHighwayGoalDeviation(const Configuration& configuration, RoadAttributes& roadAttributes) const;
	void applyNaturalPatternRule(const Configuration& configuration, const vml_vec2& position, unsigned int goalDistance, int& delay, RoadAttributes& roadAttributes, RuleAttributes& ruleAttributes) const;
	void applyRadialPatternRule(const Configuration& configuration, const vml_vec2& position, unsigned int goalDistance, int& delay, RoadAttributes& roadAttributes, RuleAttributes& ruleAttributes) const;
	void applyRasterPatternRule(const Configuration& configuration, const vml_vec2& position, unsigned int goalDistance, int& delay, RoadAttributes& roadAttributes, RuleAttributes& ruleAttributes) const;

};

#endif