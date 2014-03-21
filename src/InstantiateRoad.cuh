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

#define NONE 0
#define LEFT_BRANCH 1
#define RIGHT_BRANCH 2
#define ROAD_CONTINUATION 4

#define hasMask(x, y) (x & y) != 0
#define addMask(x, y) x != y

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE bool findHighestPopulationDensity(const vml_vec2& start, float startingAngle, vml_vec2& goal, unsigned int& distance, Context* context);
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE Pattern findUnderlyingPattern(const vml_vec2& position, Context* context);
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE bool applyNaturalPatternRule(const vml_vec2& position, unsigned int goalDistance, Highway& highway, Context* context);
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE bool applyRadialPatternRule(const vml_vec2& position, unsigned int goalDistance, Highway& highway, Context* context);
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE bool applyRasterPatternRule(const vml_vec2& position, unsigned int goalDistance, Highway& highway, Context* context);

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE unsigned int createNewWorkItems(Street& road, int source, const vml_vec2& position, Street& leftBranch, Street& rightBranch, Street& roadContinuation, Context* context)
{
	unsigned int newDepth = road.ruleAttributes.branchDepth + 1;

	if (newDepth > g_dConfiguration.maxStreetBranchDepth)
	{
		return NONE;
	}

	unsigned int creationMask = NONE;

	if (hasMask(road.ruleAttributes.expansionMask, EXPAND_UP))
	{
		creationMask |= ROAD_CONTINUATION;
		roadContinuation.roadAttributes.source = source;
		roadContinuation.roadAttributes.length = g_dConfiguration.streetLength;
		roadContinuation.roadAttributes.angle = road.roadAttributes.angle;
		roadContinuation.ruleAttributes.branchDepth = newDepth;
		roadContinuation.ruleAttributes.boundsIndex = road.ruleAttributes.boundsIndex;
		roadContinuation.ruleAttributes.expansionMask = road.ruleAttributes.expansionMask;
	}

	if (hasMask(road.ruleAttributes.expansionMask, EXPAND_RIGHT))
	{
		creationMask |= RIGHT_BRANCH;
		rightBranch.roadAttributes.source = source;
		rightBranch.roadAttributes.length = g_dConfiguration.streetLength;
		rightBranch.roadAttributes.angle = road.roadAttributes.angle + HALF_PI;
		rightBranch.ruleAttributes.branchDepth = newDepth;
		rightBranch.ruleAttributes.boundsIndex = road.ruleAttributes.boundsIndex;
		rightBranch.ruleAttributes.expansionMask = road.ruleAttributes.expansionMask;
	}

	if (hasMask(road.ruleAttributes.expansionMask, EXPAND_LEFT))
	{
		creationMask |= LEFT_BRANCH;
		leftBranch.roadAttributes.source = source;
		leftBranch.roadAttributes.length = g_dConfiguration.streetLength;
		leftBranch.roadAttributes.angle = road.roadAttributes.angle - HALF_PI;
		leftBranch.ruleAttributes.branchDepth = newDepth;
		leftBranch.ruleAttributes.boundsIndex = road.ruleAttributes.boundsIndex;
		leftBranch.ruleAttributes.expansionMask = road.ruleAttributes.expansionMask;
	}

	return creationMask;
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE unsigned int createNewWorkItems(Highway& road, int source, const vml_vec2& position, HighwayBranch& leftBranch, HighwayBranch& rightBranch, Highway& roadContinuation, Context* context)
{
	unsigned int newDepth = road.ruleAttributes.branchDepth + 1;

	if (newDepth > g_dConfiguration.maxHighwayBranchDepth)
	{
		return NONE;
	}

	bool branch = (road.ruleAttributes.branchingDistance == g_dConfiguration.highwayBranchingDistance);

	// highway continuation
	unsigned int creationMask = ROAD_CONTINUATION;

	roadContinuation.roadAttributes.source = source;
	roadContinuation.roadAttributes.length = g_dConfiguration.highwayLength;
	roadContinuation.ruleAttributes.hasGoal = road.ruleAttributes.hasGoal;
	roadContinuation.ruleAttributes.branchDepth = newDepth;
	roadContinuation.ruleAttributes.setGoal(road.ruleAttributes.getGoal());
	roadContinuation.ruleAttributes.branchingDistance = (branch) ? 0 : road.ruleAttributes.branchingDistance + 1;

	unsigned int goalDistance = 0;
	if (roadContinuation.ruleAttributes.hasGoal)
	{
		goalDistance = (unsigned int)vml_distance(position, roadContinuation.ruleAttributes.getGoal());
	}

	if (!roadContinuation.ruleAttributes.hasGoal || goalDistance <= g_dConfiguration.goalDistanceThreshold)
	{
		vml_vec2 goal;
		if (!findHighestPopulationDensity(position, road.roadAttributes.angle, goal, goalDistance, context))
		{
			return NONE;
		}

		roadContinuation.ruleAttributes.setGoal(goal);
		roadContinuation.ruleAttributes.hasGoal = true;
	}

	Pattern pattern = findUnderlyingPattern(position, context);

	if (pattern == NATURAL_PATTERN)
	{
		if (!applyNaturalPatternRule(position, goalDistance, roadContinuation, context))
		{
			return NONE;
		}
	}

	else if (pattern == RADIAL_PATTERN)
	{
		if (!applyRadialPatternRule(position, goalDistance, roadContinuation, context))
		{
			return NONE;
		}
	}

	else if (pattern == RASTER_PATTERN)
	{
		if (!applyRasterPatternRule(position, goalDistance, roadContinuation, context))
		{
			return NONE;
		}
	}

	else
	{
		// FIXME: checking invariants
		THROW_EXCEPTION("invalid pattern");
	}

	if (branch) 
	{
		// new highway branch left
		creationMask |= LEFT_BRANCH;
		leftBranch.delay = 0;
		leftBranch.roadAttributes.source = source;
		leftBranch.roadAttributes.length = g_dConfiguration.highwayLength;
		leftBranch.roadAttributes.angle = roadContinuation.roadAttributes.angle - HALF_PI;
		leftBranch.ruleAttributes.branchDepth = 0;
		leftBranch.ruleAttributes.hasGoal = false;

		// new highway branch right
		creationMask |= RIGHT_BRANCH;
		rightBranch.delay = 0;
		rightBranch.roadAttributes.source = source;
		rightBranch.roadAttributes.length = g_dConfiguration.highwayLength;
		rightBranch.roadAttributes.angle = roadContinuation.roadAttributes.angle + HALF_PI;
		rightBranch.ruleAttributes.branchDepth = 0;
		rightBranch.ruleAttributes.hasGoal = false;
	}

	return creationMask;
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE Pattern findUnderlyingPattern(const vml_vec2& position, Context* context)
{
	unsigned char naturalPattern = 0;

	if (context->naturalPatternMap != 0)
	{
		//naturalPattern = TEX2D(g_dNaturalPatternTexture, (int)position.x, (int)position.y);
		naturalPattern = TEX2D(context->naturalPatternMap, (int)position.x, (int)position.y);
	}

	unsigned char radialPattern = 0;

	if (context->radialPatternMap != 0)
	{
		//radialPattern = TEX2D(g_dRadialPatternTexture, (int)position.x, (int)position.y);
		radialPattern = TEX2D(context->radialPatternMap, (int)position.x, (int)position.y);
	}

	unsigned char rasterPattern = 0;

	if (context->rasterPatternMap != 0)
	{
		//rasterPattern = TEX2D(g_dRasterPatternTexture, (int)position.x, (int)position.y);
		rasterPattern = TEX2D(context->rasterPatternMap, (int)position.x, (int)position.y);
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
	int currentAngleStep = -g_dConfiguration.halfSamplingArc;

	unsigned char populationDensitiesSamplingBuffer[360];
	unsigned int distancesSamplingBuffer[360];

	for (unsigned int i = 0; i < g_dConfiguration.samplingArc; i++, currentAngleStep++)
	{
		vml_vec2 direction = vml_normalize(vml_rotate2D(vml_vec2(0.0f, 1.0f), startingAngle + vml_radians((float)currentAngleStep)));
		unsigned char populationDensity;
		int distance;
		//SCAN(g_dPopulationDensityTexture, start, direction, g_dConfiguration.minSamplingRayLength, g_dConfiguration.maxSamplingRayLength, populationDensity, distance);
		SCAN(context->populationDensityMap, start, direction, g_dConfiguration.minSamplingRayLength, g_dConfiguration.maxSamplingRayLength, populationDensity, distance);
		populationDensitiesSamplingBuffer[i] = populationDensity;
		distancesSamplingBuffer[i] = distance;
	}

	unsigned int highestWeight = 0;
	unsigned int j = 0;
	float angleIncrement = 0.0f;

	for (unsigned int i = 0; i < g_dConfiguration.samplingArc; i++)
	{
		unsigned int weight = populationDensitiesSamplingBuffer[i] * distancesSamplingBuffer[i];

		if (weight > highestWeight)
		{
			highestWeight = weight;
			angleIncrement = (float)i;
			j = i;
		}
	}

	if (highestWeight < g_dConfiguration.minSamplingWeight)
	{
		return false;
	}

	distance = distancesSamplingBuffer[j];
	goal = start + vml_rotate2D(vml_vec2(0.0f, (float)distance), startingAngle + vml_radians(angleIncrement - (float)g_dConfiguration.halfSamplingArc));

	return true;
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void applyGoalDeviation(const vml_vec2& position, Highway& highway, Context* context)
{
	if (g_dConfiguration.maxHighwayGoalDeviation == 0)
	{
		return;
	}

	highway.roadAttributes.angle += vml_radians((float)(RAND(position.x, position.y) % g_dConfiguration.halfMaxHighwayGoalDeviation) - (int)g_dConfiguration.maxHighwayGoalDeviation);
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE bool applyNaturalPatternRule(const vml_vec2& position, unsigned int goalDistance, Highway& highway, Context* context)
{
	highway.roadAttributes.length = MathExtras::min(goalDistance, g_dConfiguration.highwayLength);
	highway.roadAttributes.angle = MathExtras::getAngle(vml_vec2(0.0f, 1.0f), highway.ruleAttributes.getGoal() - position);
	applyGoalDeviation(position, highway, context);
	return true;
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE bool applyRadialPatternRule(const vml_vec2& position, unsigned int goalDistance, Highway& highway, Context* context)
{
	// TODO:
	return false;
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE bool applyRasterPatternRule(const vml_vec2& position, unsigned int goalDistance, Highway& highway, Context* context)
{
	float angle = MathExtras::getAngle(vml_vec2(1.0f, 0.0f), highway.ruleAttributes.getGoal() - position);
	unsigned int horizontalDistance = (unsigned int)abs((float)goalDistance * cos(angle));
	unsigned int verticalDistance = (unsigned int)abs((float)goalDistance * sin(angle));
	bool canMoveHorizontally = horizontalDistance >= g_dConfiguration.minHighwayLength;
	bool canMoveVertically = verticalDistance >= g_dConfiguration.minHighwayLength;
	bool moveHorizontally;

	if (!canMoveHorizontally && !canMoveVertically)
	{
		return false;
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

	unsigned int length = (RAND(position.x, position.y) % (g_dConfiguration.highwayLength - g_dConfiguration.minHighwayLength)) + g_dConfiguration.minHighwayLength;

	if (moveHorizontally)
	{
		highway.roadAttributes.length = MathExtras::min(horizontalDistance, length);
		highway.roadAttributes.angle = (angle > HALF_PI && angle < PI_AND_HALF) ? HALF_PI : -HALF_PI;
	}

	else
	{
		highway.roadAttributes.length = MathExtras::min(verticalDistance, length);
		highway.roadAttributes.angle = (angle > PI) ? PI : 0;
	}

	return true;
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
		bool run = addStreet(context->graph, context->primitives, road.roadAttributes.source, direction, road.ruleAttributes.boundsIndex, newSource, position);
		if (run)
		{
			Street leftBranch(UNASSIGNED);
			Street rightBranch(UNASSIGNED);
			Street roadContinuation(UNASSIGNED);

			unsigned int creationMask = createNewWorkItems(road, newSource, position, leftBranch, rightBranch, roadContinuation, context);

			if ((creationMask & LEFT_BRANCH) != 0)
			{
				backQueues[EVALUATE_STREET].push(leftBranch);
			}

			if ((creationMask & RIGHT_BRANCH) != 0)
			{
				backQueues[EVALUATE_STREET].push(rightBranch);
			}

			if ((creationMask & ROAD_CONTINUATION) != 0)
			{
				backQueues[EVALUATE_STREET].push(roadContinuation);
			}
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

		HighwayBranch leftBranch;
		HighwayBranch rightBranch;
		Highway roadContinuation(UNASSIGNED);

		unsigned int creationMask = createNewWorkItems(road, newSource, position, leftBranch, rightBranch, roadContinuation, context);

		if ((creationMask & LEFT_BRANCH) != 0)
		{
			backQueues[EVALUATE_HIGHWAY_BRANCH].push(leftBranch);
		}

		if ((creationMask & RIGHT_BRANCH) != 0)
		{
			backQueues[EVALUATE_HIGHWAY_BRANCH].push(rightBranch);
		}

		if ((creationMask & ROAD_CONTINUATION) != 0)
		{
			backQueues[EVALUATE_HIGHWAY].push(roadContinuation);
		}
	}
};

#endif