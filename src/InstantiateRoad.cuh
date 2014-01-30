#ifndef INSTANTIATEROAD_CUH
#define INSTANTIATEROAD_CUH

#pragma once

#include <CpuGpuCompatibility.h>
#include <Road.h>
#include <Branch.h>
#include <ProceduresCodes.h>
#include <Pattern.h>
#include <GlobalVariables.cuh>
#include <WorkQueue.cuh>
#include <GraphFunctions.cuh>

//#include <random>

//////////////////////////////////////////////////////////////////////////
template<typename RuleAttributesType>
DEVICE_CODE void evaluateGlobalGoals(Road<RuleAttributesType>& road, int newOrigin, const vml_vec2& position, int* delays, RoadAttributes* roadAttributes, RuleAttributesType* ruleAttributes);
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void findHighestPopulationDensity(const vml_vec2& start, float startingAngle, vml_vec2& goal, unsigned int& distance);
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE Pattern findUnderlyingPattern(const vml_vec2& position);
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void applyHighwayGoalDeviation(RoadAttributes& roadAttributes);
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void applyNaturalPatternRule(const vml_vec2& position, unsigned int goalDistance, int& delay, RoadAttributes& roadAttributes, HighwayRuleAttributes& ruleAttributes);
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void applyRadialPatternRule(const vml_vec2& position, unsigned int goalDistance, int& delay, RoadAttributes& roadAttributes, HighwayRuleAttributes& ruleAttributes);
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void applyRasterPatternRule(const vml_vec2& position, unsigned int goalDistance, int& delay, RoadAttributes& roadAttributes, HighwayRuleAttributes& ruleAttributes);

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void evaluateGlobalGoals(Street& road, int source, const vml_vec2& position, int* delays, RoadAttributes* roadAttributes, StreetRuleAttributes* ruleAttributes)
{
		unsigned int newDepth = road.ruleAttributes.branchDepth + 1;
		// street branch left
		delays[0] = g_dConfiguration->streetBranchingDelay;
		roadAttributes[0].source = source;
		roadAttributes[0].length = g_dConfiguration->streetLength;
		roadAttributes[0].angle = road.roadAttributes.angle - HALF_PI;
		ruleAttributes[0].branchDepth = newDepth;
		// street branch right
		delays[1] = g_dConfiguration->streetBranchingDelay;
		roadAttributes[1].source = source;
		roadAttributes[1].length = g_dConfiguration->streetLength;
		roadAttributes[1].angle = road.roadAttributes.angle + HALF_PI;
		ruleAttributes[1].branchDepth = newDepth;
		// street continuation
		delays[2] = 0;
		roadAttributes[2].source = source;
		roadAttributes[2].length = g_dConfiguration->streetLength;
		roadAttributes[2].angle = road.roadAttributes.angle;
		ruleAttributes[2].branchDepth = newDepth;
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void evaluateGlobalGoals(Highway& road, int source, const vml_vec2& position, int* delays, RoadAttributes* roadAttributes, HighwayRuleAttributes* ruleAttributes)
{
	bool branch = (road.ruleAttributes.branchingDistance == g_dConfiguration->minHighwayBranchingDistance);
	// highway continuation
	delays[2] = 0;
	roadAttributes[2].source = source;
	roadAttributes[2].length = g_dConfiguration->highwayLength;
	ruleAttributes[2].hasGoal = road.ruleAttributes.hasGoal;
	ruleAttributes[2].setGoal(road.ruleAttributes.getGoal());
	ruleAttributes[2].branchingDistance = (branch) ? 0 : road.ruleAttributes.branchingDistance + 1;

	unsigned int goalDistance;
	if (ruleAttributes[2].hasGoal)
	{
		goalDistance = (unsigned int)vml_distance(position, road.ruleAttributes.getGoal());
	}

	if (!ruleAttributes[2].hasGoal || goalDistance <= g_dConfiguration->goalDistanceThreshold)
	{
		vml_vec2 goal;
		findHighestPopulationDensity(position, road.roadAttributes.angle, goal, goalDistance);
		ruleAttributes[2].setGoal(goal);
		ruleAttributes[2].hasGoal = true;
	}

	Pattern pattern = findUnderlyingPattern(position);

	if (pattern == NATURAL_PATTERN)
	{
		applyNaturalPatternRule(position, goalDistance, delays[2], roadAttributes[2], ruleAttributes[2]);
	}

	else if (pattern == RADIAL_PATTERN)
	{
		applyRadialPatternRule(position, goalDistance, delays[2], roadAttributes[2], ruleAttributes[2]);
	}

	else if (pattern == RASTER_PATTERN)
	{
		applyRasterPatternRule(position, goalDistance, delays[2], roadAttributes[2], ruleAttributes[2]);
	}

	else
	{
		// FIXME: checking invariants
		THROW_EXCEPTION("invalid pattern");
	}

	if (branch) 
	{
		// new highway branch left
		delays[0] = 0;
		roadAttributes[0].source = source;
		roadAttributes[0].length = g_dConfiguration->highwayLength;
		roadAttributes[0].angle = roadAttributes[2].angle - HALF_PI;
		// new highway branch right
		delays[1] = 0;
		roadAttributes[1].source = source;
		roadAttributes[1].length = g_dConfiguration->highwayLength;
		roadAttributes[1].angle = roadAttributes[2].angle + HALF_PI;
	}
	else
	{
		delays[0] = -1;
		delays[1] = -1;
	}
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE Pattern findUnderlyingPattern(const vml_vec2& position)
{
	unsigned char naturalPattern = 0;

	if (g_dNaturalPatternMap != 0)
	{
		naturalPattern = g_dNaturalPatternMap->sample(position);
	}

	unsigned char radialPattern = 0;

	if (g_dRadialPatternMap != 0)
	{
		radialPattern = g_dRadialPatternMap->sample(position);
	}

	unsigned char rasterPattern = 0;

	if (g_dRasterPatternMap != 0)
	{
		rasterPattern = g_dRasterPatternMap->sample(position);
	}

	if (rasterPattern > radialPattern)
	{
		if (rasterPattern > naturalPattern)
		{
			return RASTER_PATTERN;
		}

		else
		{
			return NATURAL_PATTERN;
		}
	}

	else
	{
		if (radialPattern > naturalPattern)
		{
			return RADIAL_PATTERN;
		}

		else
		{
			return NATURAL_PATTERN;
		}
	}
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void findHighestPopulationDensity(const vml_vec2& start, float startingAngle, vml_vec2& goal, unsigned int& distance)
{
	int currentAngleStep = -g_dConfiguration->halfSamplingArc;

	// FIXME: checking invariants
	if (g_dPopulationDensityMap == 0)
	{
		THROW_EXCEPTION("configuration->populationDensityMap == 0");
	}

	for (unsigned int i = 0; i < g_dConfiguration->samplingArc; i++, currentAngleStep++)
	{
		vml_vec2 direction = vml_normalize(vml_rotate2D(vml_vec2(0.0f, 1.0f), startingAngle + vml_radians((float)currentAngleStep)));
		unsigned char populationDensity;
		int distance;
		g_dPopulationDensityMap->scan(start, direction, g_dConfiguration->minSamplingRayLength, g_dConfiguration->maxSamplingRayLength, populationDensity, distance);
		g_dPopulationDensitiesSamplingBuffer[i] = populationDensity;
		g_dDistancesSamplingBuffer[i] = distance;
	}

	unsigned int highestWeight = 0;
	unsigned int j = 0;
	float angleIncrement = 0.0f;

	for (unsigned int i = 0; i < g_dConfiguration->samplingArc; i++)
	{
		unsigned int weight = g_dPopulationDensitiesSamplingBuffer[i] * g_dDistancesSamplingBuffer[i];

		if (weight > highestWeight)
		{
			highestWeight = weight;
			angleIncrement = (float)i;
			j = i;
		}
	}

	distance = g_dDistancesSamplingBuffer[j];
	goal = start + vml_rotate2D(vml_vec2(0.0f, (float)distance), startingAngle + vml_radians(angleIncrement - (float)g_dConfiguration->halfSamplingArc));
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void applyHighwayGoalDeviation(RoadAttributes& roadAttributes)
{
	if (g_dConfiguration->maxHighwayGoalDeviation == 0)
	{
		return;
	}

	roadAttributes.angle += vml_radians((float)(/*rand()*/ 1123123 % g_dConfiguration->halfMaxHighwayGoalDeviation) - (int)g_dConfiguration->maxHighwayGoalDeviation);
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void applyNaturalPatternRule(const vml_vec2& position, unsigned int goalDistance, int& delay, RoadAttributes& roadAttributes, HighwayRuleAttributes& ruleAttributes)
{
	roadAttributes.length = MathExtras::min(goalDistance, g_dConfiguration->highwayLength);
	roadAttributes.angle = MathExtras::getAngle(vml_vec2(0.0f, 1.0f), ruleAttributes.getGoal() - position);
	applyHighwayGoalDeviation(roadAttributes);
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void applyRadialPatternRule(const vml_vec2& position, unsigned int goalDistance, int& delay, RoadAttributes& roadAttributes, HighwayRuleAttributes& ruleAttributes)
{
	// TODO:
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void applyRasterPatternRule(const vml_vec2& position, unsigned int goalDistance, int& delay, RoadAttributes& roadAttributes, HighwayRuleAttributes& ruleAttributes)
{
	float angle = MathExtras::getAngle(vml_vec2(1.0f, 0.0f), ruleAttributes.getGoal() - position);
	unsigned int horizontalDistance = (unsigned int)abs((float)goalDistance * cos(angle));
	unsigned int verticalDistance = (unsigned int)abs((float)goalDistance * sin(angle));
	bool canMoveHorizontally = horizontalDistance >= g_dConfiguration->minRoadLength;
	bool canMoveVertically = verticalDistance >= g_dConfiguration->minRoadLength;
	bool moveHorizontally;

	if (!canMoveHorizontally && !canMoveVertically)
	{
		delay = -1;
		return;
	}

	else if (!canMoveHorizontally)
	{
		moveHorizontally = false;
	}

	else if (!canMoveVertically)
	{
		moveHorizontally = true;
	}

	else
	{
		moveHorizontally = (/*rand()*/ 53923 % 99) < 50;
	}

	unsigned int length = (/*rand()*/ 3429753 % (g_dConfiguration->highwayLength - g_dConfiguration->minRoadLength)) + g_dConfiguration->minRoadLength;

	if (moveHorizontally)
	{
		roadAttributes.length = MathExtras::min(horizontalDistance, length);
		roadAttributes.angle = (angle > HALF_PI && angle < PI_AND_HALF) ? HALF_PI : -HALF_PI;
	}

	else
	{
		roadAttributes.length = MathExtras::min(verticalDistance, length);
		roadAttributes.angle = (angle > PI) ? PI : 0;
	}
}

//////////////////////////////////////////////////////////////////////////
struct InstantiateStreet
{
	//////////////////////////////////////////////////////////////////////////
	static DEVICE_CODE void execute(Street& road, WorkQueue* backQueues)
	{
		// p2

		// FIXME: checking invariants
		if (road.state != SUCCEED)
		{
			THROW_EXCEPTION("road.state != SUCCEED");
		}

		vml_vec2 direction = vml_rotate2D(vml_vec2(0.0f, road.roadAttributes.length), road.roadAttributes.angle);
		int newSource;
		vml_vec2 position;
		bool interrupted = addRoad(g_dGraph, road.roadAttributes.source, direction, newSource, position, false);
		int delays[3];
		RoadAttributes roadAttributes[3];
		StreetRuleAttributes ruleAttributes[3];

		if (interrupted)
		{
			delays[0] = -1;
			delays[1] = -1;
			delays[2] = -1;
		}

		else
		{
			evaluateGlobalGoals(road, newSource, position, delays, roadAttributes, ruleAttributes);
		}

		backQueues[EVALUATE_STREET_BRANCH].push(Branch<StreetRuleAttributes>(delays[0], roadAttributes[0], ruleAttributes[0]));
		backQueues[EVALUATE_STREET_BRANCH].push(Branch<StreetRuleAttributes>(delays[1], roadAttributes[1], ruleAttributes[1]));
		backQueues[EVALUATE_STREET].push(Street(delays[2], roadAttributes[2], ruleAttributes[2], UNASSIGNED));
	}
};

//////////////////////////////////////////////////////////////////////////
struct InstantiateHighway
{
	//////////////////////////////////////////////////////////////////////////
	static DEVICE_CODE void execute(Highway& road, WorkQueue* backQueues)
	{
		// p2

		// FIXME: checking invariants
		if (road.state != SUCCEED)
		{
			THROW_EXCEPTION("road.state != SUCCEED");
		}

		vml_vec2 direction = vml_rotate2D(vml_vec2(0.0f, road.roadAttributes.length), road.roadAttributes.angle);
		int newSource;
		vml_vec2 position;
		bool interrupted = addRoad(g_dGraph, road.roadAttributes.source, direction, newSource, position, true);
		int delays[3];
		RoadAttributes roadAttributes[3];
		HighwayRuleAttributes ruleAttributes[3];

		if (interrupted)
		{
			delays[0] = -1;
			delays[1] = -1;
			delays[2] = -1;
		}

		else
		{
			evaluateGlobalGoals(road, newSource, position, delays, roadAttributes, ruleAttributes);
		}

		backQueues[EVALUATE_HIGHWAY_BRANCH].push(Branch<HighwayRuleAttributes>(delays[0], roadAttributes[0], ruleAttributes[0]));
		backQueues[EVALUATE_HIGHWAY_BRANCH].push(Branch<HighwayRuleAttributes>(delays[1], roadAttributes[1], ruleAttributes[1]));
		backQueues[EVALUATE_HIGHWAY].push(Highway(delays[2], roadAttributes[2], ruleAttributes[2], UNASSIGNED));
	}
};

#endif