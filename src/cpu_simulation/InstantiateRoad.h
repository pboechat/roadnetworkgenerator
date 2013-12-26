#ifndef INSTANTIATEROAD_H
#define INSTANTIATEROAD_H

#include <Procedures.h>
#include <Road.h>
#include <Pattern.h>

#include <vector_math.h>

struct InstantiateRoad : public Procedure<InstantiateRoad, Road, 2>
{
	static void execute(Road& road, WorkQueues& manager, RoadNetworkGraph::Graph& graph, const Configuration& configuration);

	// TODO: improve design
	static void initialize(const Configuration& configuration);
	static void dispose();

private:
	static unsigned char* populationDensities;
	static unsigned int* distances;

	static void evaluateGlobalGoals(Road& road, const Configuration& configuration, RoadNetworkGraph::VertexIndex newOrigin, const vml_vec2& position, int* delays, RoadAttributes* roadAttributes, RuleAttributes* ruleAttributes);
	static void findHighestPopulationDensity(const Configuration& configuration, const vml_vec2& start, float startingAngle, vml_vec2& goal, unsigned int& distance) const;
	static Pattern findUnderlyingPattern(const Configuration& configuration, const vml_vec2& position) const;
	static void applyHighwayGoalDeviation(const Configuration& configuration, RoadAttributes& roadAttributes) const;
	static void applyNaturalPatternRule(const Configuration& configuration, const vml_vec2& position, unsigned int goalDistance, int& delay, RoadAttributes& roadAttributes, RuleAttributes& ruleAttributes) const;
	static void applyRadialPatternRule(const Configuration& configuration, const vml_vec2& position, unsigned int goalDistance, int& delay, RoadAttributes& roadAttributes, RuleAttributes& ruleAttributes) const;
	static void applyRasterPatternRule(const Configuration& configuration, const vml_vec2& position, unsigned int goalDistance, int& delay, RoadAttributes& roadAttributes, RuleAttributes& ruleAttributes) const;

};

#endif