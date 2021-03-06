#include <ProceduresDeclarations.h>
#include <ProceduresCodes.h>
#include <Globals.h>
#include <Pattern.h>

#include <random>

//////////////////////////////////////////////////////////////////////////
template<typename RuleAttributesType>
void evaluateGlobalGoals(Road<RuleAttributesType>& road, RoadNetworkGraph::VertexIndex newOrigin, const vml_vec2& position, int* delays, RoadAttributes* roadAttributes, RuleAttributesType* ruleAttributes);
//////////////////////////////////////////////////////////////////////////
void findHighestPopulationDensity(const vml_vec2& start, float startingAngle, vml_vec2& goal, unsigned int& distance);
//////////////////////////////////////////////////////////////////////////
Pattern findUnderlyingPattern(const vml_vec2& position);
//////////////////////////////////////////////////////////////////////////
void applyHighwayGoalDeviation(RoadAttributes& roadAttributes);
//////////////////////////////////////////////////////////////////////////
void applyNaturalPatternRule(const vml_vec2& position, unsigned int goalDistance, int& delay, RoadAttributes& roadAttributes, HighwayRuleAttributes& ruleAttributes);
//////////////////////////////////////////////////////////////////////////
void applyRadialPatternRule(const vml_vec2& position, unsigned int goalDistance, int& delay, RoadAttributes& roadAttributes, HighwayRuleAttributes& ruleAttributes);
//////////////////////////////////////////////////////////////////////////
void applyRasterPatternRule(const vml_vec2& position, unsigned int goalDistance, int& delay, RoadAttributes& roadAttributes, HighwayRuleAttributes& ruleAttributes);

//////////////////////////////////////////////////////////////////////////
void InstantiateStreet::execute(Street& road, WorkQueuesSet* backQueues)
{
	// p2

	// FIXME: checking invariants
	if (road.state != SUCCEED)
	{
		throw std::exception("road.state != SUCCEED");
	}

	vml_vec2 direction = vml_rotate2D(vml_vec2(0.0f, road.roadAttributes.length), road.roadAttributes.angle);
	RoadNetworkGraph::VertexIndex newSource;
	vml_vec2 position;
	bool interrupted = addRoad(g_graph, road.roadAttributes.source, direction, newSource, position, false);
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

	backQueues->addWorkItem(EVALUATE_STREET_BRANCH, Branch<StreetRuleAttributes>(delays[0], roadAttributes[0], ruleAttributes[0]));
	backQueues->addWorkItem(EVALUATE_STREET_BRANCH, Branch<StreetRuleAttributes>(delays[1], roadAttributes[1], ruleAttributes[1]));
	backQueues->addWorkItem(EVALUATE_STREET, Street(delays[2], roadAttributes[2], ruleAttributes[2], UNASSIGNED));
}

//////////////////////////////////////////////////////////////////////////
void InstantiateHighway::execute(Highway& road, WorkQueuesSet* backQueues)
{
	// p2

	// FIXME: checking invariants
	if (road.state != SUCCEED)
	{
		throw std::exception("road.state != SUCCEED");
	}

	vml_vec2 direction = vml_rotate2D(vml_vec2(0.0f, road.roadAttributes.length), road.roadAttributes.angle);
	RoadNetworkGraph::VertexIndex newSource;
	vml_vec2 position;
	bool interrupted = addRoad(g_graph, road.roadAttributes.source, direction, newSource, position, true);
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

	backQueues->addWorkItem(EVALUATE_HIGHWAY_BRANCH, Branch<HighwayRuleAttributes>(delays[0], roadAttributes[0], ruleAttributes[0]));
	backQueues->addWorkItem(EVALUATE_HIGHWAY_BRANCH, Branch<HighwayRuleAttributes>(delays[1], roadAttributes[1], ruleAttributes[1]));
	backQueues->addWorkItem(EVALUATE_HIGHWAY, Highway(delays[2], roadAttributes[2], ruleAttributes[2], UNASSIGNED));
}

//////////////////////////////////////////////////////////////////////////
void evaluateGlobalGoals(Street& road, RoadNetworkGraph::VertexIndex source, const vml_vec2& position, int* delays, RoadAttributes* roadAttributes, StreetRuleAttributes* ruleAttributes)
{
		unsigned int newDepth = road.ruleAttributes.branchDepth + 1;
		// street branch left
		delays[0] = g_configuration->streetBranchingDelay;
		roadAttributes[0].source = source;
		roadAttributes[0].length = g_configuration->streetLength;
		roadAttributes[0].angle = road.roadAttributes.angle - MathExtras::HALF_PI;
		ruleAttributes[0].branchDepth = newDepth;
		// street branch right
		delays[1] = g_configuration->streetBranchingDelay;
		roadAttributes[1].source = source;
		roadAttributes[1].length = g_configuration->streetLength;
		roadAttributes[1].angle = road.roadAttributes.angle + MathExtras::HALF_PI;
		ruleAttributes[1].branchDepth = newDepth;
		// street continuation
		delays[2] = 0;
		roadAttributes[2].source = source;
		roadAttributes[2].length = g_configuration->streetLength;
		roadAttributes[2].angle = road.roadAttributes.angle;
		ruleAttributes[2].branchDepth = newDepth;
}

//////////////////////////////////////////////////////////////////////////
void evaluateGlobalGoals(Highway& road, RoadNetworkGraph::VertexIndex source, const vml_vec2& position, int* delays, RoadAttributes* roadAttributes, HighwayRuleAttributes* ruleAttributes)
{
	bool branch = (road.ruleAttributes.branchingDistance == g_configuration->minHighwayBranchingDistance);
	// highway continuation
	delays[2] = 0;
	roadAttributes[2].source = source;
	ruleAttributes[2].hasGoal = road.ruleAttributes.hasGoal;
	ruleAttributes[2].goal = road.ruleAttributes.goal;
	ruleAttributes[2].branchingDistance = (branch) ? 0 : road.ruleAttributes.branchingDistance + 1;

	unsigned int goalDistance;
	if (ruleAttributes[2].hasGoal)
	{
		goalDistance = (unsigned int)vml_distance(position, road.ruleAttributes.goal);
	}

	if (!ruleAttributes[2].hasGoal || goalDistance <= g_configuration->goalDistanceThreshold)
	{
		findHighestPopulationDensity(position, road.roadAttributes.angle, ruleAttributes[2].goal, goalDistance);
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
		throw std::exception("invalid pattern");
	}

	if (branch) 
	{
		// new highway branch left
		delays[0] = 0;
		roadAttributes[0].source = source;
		roadAttributes[0].length = g_configuration->highwayLength;
		roadAttributes[0].angle = roadAttributes[2].angle - MathExtras::HALF_PI;
		// new highway branch right
		delays[1] = 0;
		roadAttributes[1].source = source;
		roadAttributes[1].length = g_configuration->highwayLength;
		roadAttributes[1].angle = roadAttributes[2].angle + MathExtras::HALF_PI;
	}
}

//////////////////////////////////////////////////////////////////////////
Pattern findUnderlyingPattern(const vml_vec2& position)
{
	unsigned char naturalPattern = 0;

	if (g_naturalPatternMap != 0)
	{
		naturalPattern = g_naturalPatternMap->sample(position);
	}

	unsigned char radialPattern = 0;

	if (g_radialPatternMap != 0)
	{
		radialPattern = g_radialPatternMap->sample(position);
	}

	unsigned char rasterPattern = 0;

	if (g_rasterPatternMap != 0)
	{
		rasterPattern = g_rasterPatternMap->sample(position);
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
void findHighestPopulationDensity(const vml_vec2& start, float startingAngle, vml_vec2& goal, unsigned int& distance)
{
	int currentAngleStep = -g_configuration->halfSamplingArc;

	// FIXME: checking invariants
	if (g_populationDensityMap == 0)
	{
		throw std::exception("configuration->populationDensityMap == 0");
	}

	for (unsigned int i = 0; i < g_configuration->samplingArc; i++, currentAngleStep++)
	{
		vml_vec2 direction = vml_normalize(vml_rotate2D(vml_vec2(0.0f, 1.0f), startingAngle + vml_radians((float)currentAngleStep)));
		unsigned char populationDensity;
		int distance;
		g_populationDensityMap->scan(start, direction, g_configuration->minSamplingRayLength, g_configuration->maxSamplingRayLength, populationDensity, distance);
		g_populationDensitiesSamplingBuffer[i] = populationDensity;
		g_distancesSamplingBuffer[i] = distance;
	}

	unsigned int highestWeight = 0;
	unsigned int j = 0;
	float angleIncrement = 0.0f;

	for (unsigned int i = 0; i < g_configuration->samplingArc; i++)
	{
		unsigned int weight = g_populationDensitiesSamplingBuffer[i] * g_distancesSamplingBuffer[i];

		if (weight > highestWeight)
		{
			highestWeight = weight;
			angleIncrement = (float)i;
			j = i;
		}
	}

	distance = g_distancesSamplingBuffer[j];
	goal = start + vml_rotate2D(vml_vec2(0.0f, (float)distance), startingAngle + vml_radians(angleIncrement - (float)g_configuration->halfSamplingArc));
}

//////////////////////////////////////////////////////////////////////////
void applyHighwayGoalDeviation(RoadAttributes& roadAttributes)
{
	if (g_configuration->maxHighwayGoalDeviation == 0)
	{
		return;
	}

	roadAttributes.angle += vml_radians((float)(rand() % g_configuration->halfMaxHighwayGoalDeviation) - (int)g_configuration->maxHighwayGoalDeviation);
}

//////////////////////////////////////////////////////////////////////////
void applyNaturalPatternRule(const vml_vec2& position, unsigned int goalDistance, int& delay, RoadAttributes& roadAttributes, HighwayRuleAttributes& ruleAttributes)
{
	roadAttributes.length = MathExtras::min(goalDistance, g_configuration->highwayLength);
	roadAttributes.angle = MathExtras::getAngle(vml_vec2(0.0f, 1.0f), ruleAttributes.goal - position);
	applyHighwayGoalDeviation(roadAttributes);
}

//////////////////////////////////////////////////////////////////////////
void applyRadialPatternRule(const vml_vec2& position, unsigned int goalDistance, int& delay, RoadAttributes& roadAttributes, HighwayRuleAttributes& ruleAttributes)
{
	// TODO:
}

//////////////////////////////////////////////////////////////////////////
void applyRasterPatternRule(const vml_vec2& position, unsigned int goalDistance, int& delay, RoadAttributes& roadAttributes, HighwayRuleAttributes& ruleAttributes)
{
	float angle = MathExtras::getAngle(vml_vec2(1.0f, 0.0f), ruleAttributes.goal - position);
	unsigned int horizontalDistance = (unsigned int)abs((float)goalDistance * cos(angle));
	unsigned int verticalDistance = (unsigned int)abs((float)goalDistance * sin(angle));
	bool canMoveHorizontally = horizontalDistance >= g_configuration->minRoadLength;
	bool canMoveVertically = verticalDistance >= g_configuration->minRoadLength;
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
		moveHorizontally = (rand() % 99) < 50;
	}

	unsigned int length = (rand() % (g_configuration->highwayLength - g_configuration->minRoadLength)) + g_configuration->minRoadLength;

	if (moveHorizontally)
	{
		roadAttributes.length = MathExtras::min(horizontalDistance, length);
		roadAttributes.angle = (angle > MathExtras::HALF_PI && angle < MathExtras::PI_AND_HALF) ? MathExtras::HALF_PI : -MathExtras::HALF_PI;
	}

	else
	{
		roadAttributes.length = MathExtras::min(verticalDistance, length);
		roadAttributes.angle = (angle > MathExtras::PI) ? MathExtras::PI : 0;
	}
}