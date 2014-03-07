#ifndef INSTANTIATEROAD_CUH
#define INSTANTIATEROAD_CUH

#pragma once

#include <CpuGpuCompatibility.h>
#include <Road.h>
#include <Branch.h>
#include <ProceduresCodes.h>
#include <Pattern.h>
#include <Context.cuh>
#include <WorkQueue.cuh>
#include <GraphFunctions.cuh>
#include <PseudoRandomNumbers.cuh>
#include <GlobalVariables.cuh>

#define hasMask(x, y) (x & y) != 0
#define addMask(x, y) x != y

//////////////////////////////////////////////////////////////////////////
template<typename RuleAttributesType>
DEVICE_CODE void evaluateGlobalGoals(Road<RuleAttributesType>& road, int newOrigin, const vml_vec2& position, int* delays, RoadAttributes* roadAttributes, RuleAttributesType* ruleAttributes, Context* context);
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE bool findHighestPopulationDensity(const vml_vec2& start, float startingAngle, vml_vec2& goal, unsigned int& distance, Context* context);
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE Pattern findUnderlyingPattern(const vml_vec2& position, Context* context);
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void applyNaturalPatternRule(const vml_vec2& position, unsigned int goalDistance, int& delay, RoadAttributes& roadAttributes, HighwayRuleAttributes& ruleAttributes, Context* context);
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void applyRadialPatternRule(const vml_vec2& position, unsigned int goalDistance, int& delay, RoadAttributes& roadAttributes, HighwayRuleAttributes& ruleAttributes, Context* context);
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void applyRasterPatternRule(const vml_vec2& position, unsigned int goalDistance, int& delay, RoadAttributes& roadAttributes, HighwayRuleAttributes& ruleAttributes, Context* context);

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void evaluateGlobalGoals(Street& road, int source, const vml_vec2& position, int* delays, RoadAttributes* roadAttributes, StreetRuleAttributes* ruleAttributes, Context* context)
{
	unsigned int newDepth = road.ruleAttributes.branchDepth + 1;

	if (newDepth > context->configuration->maxStreetBranchDepth)
	{
		delays[0] = -1;
		delays[1] = -1;
		delays[2] = -1;
		return;
	}

	// left child
	delays[0] = 0;
	roadAttributes[0].source = source;
	roadAttributes[0].length = context->configuration->streetLength;
	roadAttributes[0].angle = road.roadAttributes.angle - HALF_PI;
	ruleAttributes[0].branchDepth = newDepth;
	ruleAttributes[0].boundsIndex = road.ruleAttributes.boundsIndex;
	ruleAttributes[0].childCode = LEFT_CHILD;

	// right child
	delays[1] = -1;

	// up child
	delays[2] = 0;
	roadAttributes[2].source = source;
	roadAttributes[2].length = context->configuration->streetLength;
	roadAttributes[2].angle = road.roadAttributes.angle;
	ruleAttributes[2].branchDepth = newDepth;
	ruleAttributes[2].boundsIndex = road.ruleAttributes.boundsIndex;
	ruleAttributes[2].childCode = UP_CHILD;
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void evaluateGlobalGoals(Highway& road, int source, const vml_vec2& position, int* delays, RoadAttributes* roadAttributes, HighwayRuleAttributes* ruleAttributes, Context* context)
{
	unsigned int newDepth = road.ruleAttributes.branchDepth + 1;

	if (newDepth > context->configuration->maxHighwayBranchDepth)
	{
		delays[0] = -1;
		delays[1] = -1;
		delays[2] = -1;
		return;
	}

	bool branch = (road.ruleAttributes.branchingDistance == context->configuration->highwayBranchingDistance);
	// highway continuation
	delays[2] = 0;
	roadAttributes[2].source = source;
	roadAttributes[2].length = context->configuration->highwayLength;
	ruleAttributes[2].hasGoal = road.ruleAttributes.hasGoal;
	ruleAttributes[2].branchDepth = newDepth;
	ruleAttributes[2].setGoal(road.ruleAttributes.getGoal());
	ruleAttributes[2].branchingDistance = (branch) ? 0 : road.ruleAttributes.branchingDistance + 1;

	unsigned int goalDistance;
	if (ruleAttributes[2].hasGoal)
	{
		goalDistance = (unsigned int)vml_distance(position, road.ruleAttributes.getGoal());
	}

	if (!ruleAttributes[2].hasGoal || goalDistance <= context->configuration->goalDistanceThreshold)
	{
		vml_vec2 goal;
		if (!findHighestPopulationDensity(position, road.roadAttributes.angle, goal, goalDistance, context))
		{
			delays[0] = -1;
			delays[1] = -1;
			delays[2] = -1;
			return;
		}
		ruleAttributes[2].setGoal(goal);
		ruleAttributes[2].hasGoal = true;
	}

	Pattern pattern = findUnderlyingPattern(position, context);

	if (pattern == NATURAL_PATTERN)
	{
		applyNaturalPatternRule(position, goalDistance, delays[2], roadAttributes[2], ruleAttributes[2], context);
	}

	else if (pattern == RADIAL_PATTERN)
	{
		applyRadialPatternRule(position, goalDistance, delays[2], roadAttributes[2], ruleAttributes[2], context);
	}

	else if (pattern == RASTER_PATTERN)
	{
		applyRasterPatternRule(position, goalDistance, delays[2], roadAttributes[2], ruleAttributes[2], context);
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
		roadAttributes[0].length = context->configuration->highwayLength;
		roadAttributes[0].angle = roadAttributes[2].angle - HALF_PI;
		ruleAttributes[0].branchDepth = 0;
		ruleAttributes[0].hasGoal = false;
		// new highway branch right
		delays[1] = 0;
		roadAttributes[1].source = source;
		roadAttributes[1].length = context->configuration->highwayLength;
		roadAttributes[1].angle = roadAttributes[2].angle + HALF_PI;
		ruleAttributes[1].branchDepth = 0;
		ruleAttributes[1].hasGoal = false;
	}
	else
	{
		delays[0] = -1;
		delays[1] = -1;
	}
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE Pattern findUnderlyingPattern(const vml_vec2& position, Context* context)
{
	unsigned char naturalPattern = 0;

	if (context->naturalPatternMap != 0)
	{
		naturalPattern = TEX2D(naturalPatternTexture, (int)position.x, (int)position.y);
	}

	unsigned char radialPattern = 0;

	if (context->radialPatternMap != 0)
	{
		radialPattern = TEX2D(radialPatternTexture, (int)position.x, (int)position.y);
	}

	unsigned char rasterPattern = 0;

	if (context->rasterPatternMap != 0)
	{
		rasterPattern = TEX2D(rasterPatternTexture, (int)position.x, (int)position.y);
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
DEVICE_CODE bool findHighestPopulationDensity(const vml_vec2& start, float startingAngle, vml_vec2& goal, unsigned int& distance, Context* context)
{
	int currentAngleStep = -context->configuration->halfSamplingArc;

	unsigned char populationDensitiesSamplingBuffer[360];
	unsigned int distancesSamplingBuffer[360];

	for (unsigned int i = 0; i < context->configuration->samplingArc; i++, currentAngleStep++)
	{
		vml_vec2 direction = vml_normalize(vml_rotate2D(vml_vec2(0.0f, 1.0f), startingAngle + vml_radians((float)currentAngleStep)));
		unsigned char populationDensity;
		int distance;
		SCAN(populationDensityTexture, start, direction, context->configuration->minSamplingRayLength, context->configuration->maxSamplingRayLength, populationDensity, distance);
		populationDensitiesSamplingBuffer[i] = populationDensity;
		distancesSamplingBuffer[i] = distance;
	}

	unsigned int highestWeight = 0;
	unsigned int j = 0;
	float angleIncrement = 0.0f;

	for (unsigned int i = 0; i < context->configuration->samplingArc; i++)
	{
		unsigned int weight = populationDensitiesSamplingBuffer[i] * distancesSamplingBuffer[i];

		if (weight > highestWeight)
		{
			highestWeight = weight;
			angleIncrement = (float)i;
			j = i;
		}
	}

	if (highestWeight < context->configuration->minSamplingWeight)
	{
		return false;
	}

	distance = distancesSamplingBuffer[j];
	goal = start + vml_rotate2D(vml_vec2(0.0f, (float)distance), startingAngle + vml_radians(angleIncrement - (float)context->configuration->halfSamplingArc));

	return true;
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void applyHighwayGoalDeviation(const vml_vec2& position, RoadAttributes& roadAttributes, Context* context)
{
	if (context->configuration->maxHighwayGoalDeviation == 0)
	{
		return;
	}

	roadAttributes.angle += vml_radians((float)(RAND(position.x, position.y) % context->configuration->halfMaxHighwayGoalDeviation) - (int)context->configuration->maxHighwayGoalDeviation);
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void applyNaturalPatternRule(const vml_vec2& position, unsigned int goalDistance, int& delay, RoadAttributes& roadAttributes, HighwayRuleAttributes& ruleAttributes, Context* context)
{
	roadAttributes.length = MathExtras::min(goalDistance, context->configuration->highwayLength);
	roadAttributes.angle = MathExtras::getAngle(vml_vec2(0.0f, 1.0f), ruleAttributes.getGoal() - position);
	applyHighwayGoalDeviation(position, roadAttributes, context);
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void applyRadialPatternRule(const vml_vec2& position, unsigned int goalDistance, int& delay, RoadAttributes& roadAttributes, HighwayRuleAttributes& ruleAttributes, Context* context)
{
	// TODO:
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void applyRasterPatternRule(const vml_vec2& position, unsigned int goalDistance, int& delay, RoadAttributes& roadAttributes, HighwayRuleAttributes& ruleAttributes, Context* context)
{
	float angle = MathExtras::getAngle(vml_vec2(1.0f, 0.0f), ruleAttributes.getGoal() - position);
	unsigned int horizontalDistance = (unsigned int)abs((float)goalDistance * cos(angle));
	unsigned int verticalDistance = (unsigned int)abs((float)goalDistance * sin(angle));
	bool canMoveHorizontally = horizontalDistance >= context->configuration->minRoadLength;
	bool canMoveVertically = verticalDistance >= context->configuration->minRoadLength;
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
		moveHorizontally = (RAND(position.x, position.y) % 99) < 50;
	}

	unsigned int length = (RAND(position.x, position.y) % (context->configuration->highwayLength - context->configuration->minRoadLength)) + context->configuration->minRoadLength;

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
	static DEVICE_CODE void execute(Street& road, Context* context, WorkQueue* backQueues)
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
		bool interrupted = addStreet(context->graph, context->primitives, road.roadAttributes.source, direction, road.ruleAttributes.boundsIndex, road.ruleAttributes.childCode, newSource, position);

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
			evaluateGlobalGoals(road, newSource, position, delays, roadAttributes, ruleAttributes, context);
		}

		if (delays[0] >= 0)
		{
			backQueues[EVALUATE_STREET].push(Street(delays[0], roadAttributes[0], ruleAttributes[0], UNASSIGNED));
		}

		if (delays[1] >= 0)
		{
			backQueues[EVALUATE_STREET].push(Street(delays[1], roadAttributes[1], ruleAttributes[1], UNASSIGNED));
		}

		if (delays[2] >= 0)
		{
			backQueues[EVALUATE_STREET].push(Street(delays[2], roadAttributes[2], ruleAttributes[2], UNASSIGNED));
		}
	}
};

//////////////////////////////////////////////////////////////////////////
struct InstantiateHighway
{
	//////////////////////////////////////////////////////////////////////////
	static DEVICE_CODE void execute(Highway& road, Context* context, WorkQueue* backQueues)
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
		addHighway(context->graph, road.roadAttributes.source, direction, newSource, position);

		int delays[3];
		RoadAttributes roadAttributes[3];
		HighwayRuleAttributes ruleAttributes[3];

		evaluateGlobalGoals(road, newSource, position, delays, roadAttributes, ruleAttributes, context);

		if (delays[0] >= 0)
		{
			backQueues[EVALUATE_HIGHWAY_BRANCH].push(Branch<HighwayRuleAttributes>(delays[0], roadAttributes[0], ruleAttributes[0]));
		}

		if (delays[1] >= 0)
		{
			backQueues[EVALUATE_HIGHWAY_BRANCH].push(Branch<HighwayRuleAttributes>(delays[1], roadAttributes[1], ruleAttributes[1]));
		}

		if (delays[2] >= 0)
		{
			backQueues[EVALUATE_HIGHWAY].push(Highway(delays[2], roadAttributes[2], ruleAttributes[2], UNASSIGNED));
		}
	}
};

#endif