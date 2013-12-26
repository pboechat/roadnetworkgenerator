#include <Procedures.h>
#include <Pattern.h>

#include <sstream>
#include <random>

//////////////////////////////////////////////////////////////////////////
void evaluateGlobalGoals(Road& road, RoadNetworkGraph::VertexIndex newOrigin, const vml_vec2& position, int* delays, RoadAttributes* roadAttributes, RuleAttributes* ruleAttributes);
//////////////////////////////////////////////////////////////////////////
void findHighestPopulationDensity(const vml_vec2& start, float startingAngle, vml_vec2& goal, unsigned int& distance);
//////////////////////////////////////////////////////////////////////////
Pattern findUnderlyingPattern(const vml_vec2& position);
//////////////////////////////////////////////////////////////////////////
void applyHighwayGoalDeviation(RoadAttributes& roadAttributes);
//////////////////////////////////////////////////////////////////////////
void applyNaturalPatternRule(const vml_vec2& position, unsigned int goalDistance, int& delay, RoadAttributes& roadAttributes, RuleAttributes& ruleAttributes);
//////////////////////////////////////////////////////////////////////////
void applyRadialPatternRule(const vml_vec2& position, unsigned int goalDistance, int& delay, RoadAttributes& roadAttributes, RuleAttributes& ruleAttributes);
//////////////////////////////////////////////////////////////////////////
void applyRasterPatternRule(const vml_vec2& position, unsigned int goalDistance, int& delay, RoadAttributes& roadAttributes, RuleAttributes& ruleAttributes);

//////////////////////////////////////////////////////////////////////////
void InstantiateRoad::execute(Road& road, WorkQueues* backQueues)
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
	bool interrupted = graph->addRoad(road.roadAttributes.source, direction, newSource, position, road.roadAttributes.highway);
	int delays[3];
	RoadAttributes roadAttributes[3];
	RuleAttributes ruleAttributes[3];

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

	backQueues->addWorkItem(EVALUATE_BRANCH, Branch(delays[0], roadAttributes[0], ruleAttributes[0]));
	backQueues->addWorkItem(EVALUATE_BRANCH, Branch(delays[1], roadAttributes[1], ruleAttributes[1]));
	backQueues->addWorkItem(EVALUATE_ROAD, Road(delays[2], roadAttributes[2], ruleAttributes[2], UNASSIGNED));
}

//////////////////////////////////////////////////////////////////////////
void evaluateGlobalGoals(Road& road, RoadNetworkGraph::VertexIndex source, const vml_vec2& position, int* delays, RoadAttributes* roadAttributes, RuleAttributes* ruleAttributes)
{
	if (road.roadAttributes.highway)
	{
		bool doPureHighwayBranch = (road.ruleAttributes.pureHighwayBranchingDistance == configuration->minPureHighwayBranchingDistance);
		bool doRegularBranch = (road.ruleAttributes.highwayBranchingDistance == configuration->minHighwayBranchingDistance);
		// highway continuation
		delays[2] = 0;
		roadAttributes[2].source = source;
		roadAttributes[2].highway = true;
		ruleAttributes[2].hasGoal = road.ruleAttributes.hasGoal;
		ruleAttributes[2].goal = road.ruleAttributes.goal;
		ruleAttributes[2].highwayBranchingDistance = (doRegularBranch) ? 0 : road.ruleAttributes.highwayBranchingDistance + 1;
		ruleAttributes[2].pureHighwayBranchingDistance = (doPureHighwayBranch) ? 0 : road.ruleAttributes.pureHighwayBranchingDistance + 1;
		unsigned int goalDistance;

		if (!ruleAttributes[2].hasGoal)
		{
			findHighestPopulationDensity(position, road.roadAttributes.angle, ruleAttributes[2].goal, goalDistance);
			ruleAttributes[2].hasGoal = true;
		}

		else
		{
			goalDistance = (unsigned int)vml_distance(position, road.ruleAttributes.goal);
		}

		if (goalDistance <= configuration->goalDistanceThreshold)
		{
			delays[2] = -1; // remove highway
		}

		else
		{
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
		}

		if (doPureHighwayBranch)
		{
			// new highway branch left
			delays[0] = 0;
			roadAttributes[0].source = source;
			roadAttributes[0].length = configuration->highwayLength;
			roadAttributes[0].angle = roadAttributes[2].angle - MathExtras::HALF_PI;
			roadAttributes[0].highway = true;
			// new highway branch right
			delays[1] = 0;
			roadAttributes[1].source = source;
			roadAttributes[1].length = configuration->highwayLength;
			roadAttributes[1].angle = roadAttributes[2].angle + MathExtras::HALF_PI;
			roadAttributes[1].highway = true;
		}

		else
		{
			// new street branch left
			delays[0] = (doRegularBranch) ? configuration->highwayBranchingDelay : -1;
			roadAttributes[0].source = source;
			roadAttributes[0].length = configuration->streetLength;
			roadAttributes[0].angle = roadAttributes[2].angle - MathExtras::HALF_PI;
			roadAttributes[0].highway = false;
			// new street branch right
			delays[1] = (doRegularBranch) ? configuration->highwayBranchingDelay : -1;
			roadAttributes[1].source = source;
			roadAttributes[1].length = configuration->streetLength;
			roadAttributes[1].angle = roadAttributes[2].angle + MathExtras::HALF_PI;
			roadAttributes[1].highway = false;
		}
	}

	else
	{
		unsigned int newStreetDepth = road.ruleAttributes.streetBranchDepth + 1;
		// street branch left
		delays[0] = configuration->streetBranchingDelay;
		roadAttributes[0].source = source;
		roadAttributes[0].length = configuration->streetLength;
		roadAttributes[0].angle = road.roadAttributes.angle - MathExtras::HALF_PI;
		roadAttributes[0].highway = false;
		ruleAttributes[0].streetBranchDepth = newStreetDepth;
		// street branch right
		delays[1] = configuration->streetBranchingDelay;
		roadAttributes[1].source = source;
		roadAttributes[1].length = configuration->streetLength;
		roadAttributes[1].angle = road.roadAttributes.angle + MathExtras::HALF_PI;
		roadAttributes[1].highway = false;
		ruleAttributes[1].streetBranchDepth = newStreetDepth;
		// street continuation
		delays[2] = 0;
		roadAttributes[2].source = source;
		roadAttributes[2].length = configuration->streetLength;
		roadAttributes[2].angle = road.roadAttributes.angle;
		roadAttributes[2].highway = false;
		ruleAttributes[2].streetBranchDepth = newStreetDepth;
	}
}

//////////////////////////////////////////////////////////////////////////
Pattern findUnderlyingPattern(const vml_vec2& position)
{
	unsigned char naturalPattern = 0;

	if (configuration->naturalPatternMap != 0)
	{
		naturalPattern = configuration->naturalPatternMap->sample(position);
	}

	unsigned char radialPattern = 0;

	if (configuration->radialPatternMap != 0)
	{
		radialPattern = configuration->radialPatternMap->sample(position);
	}

	unsigned char rasterPattern = 0;

	if (configuration->rasterPatternMap != 0)
	{
		rasterPattern = configuration->rasterPatternMap->sample(position);
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
	int currentAngleStep = -configuration->halfSamplingArc;

	// FIXME: checking invariants
	if (configuration->populationDensityMap == 0)
	{
		throw std::exception("configuration->populationDensityMap == 0");
	}

	for (unsigned int i = 0; i < configuration->samplingArc; i++, currentAngleStep++)
	{
		vml_vec2 direction = vml_normalize(vml_rotate2D(vml_vec2(0.0f, 1.0f), startingAngle + vml_radians((float)currentAngleStep)));
		unsigned char populationDensity;
		int distance;
		configuration->populationDensityMap->scan(start, direction, configuration->minSamplingRayLength, configuration->maxSamplingRayLength, populationDensity, distance);
		populationDensitiesSamplingBuffer[i] = populationDensity;
		distancesSamplingBuffer[i] = distance;
	}

	unsigned int highestWeight = 0;
	unsigned int j = 0;
	float angleIncrement = 0.0f;

	for (unsigned int i = 0; i < configuration->samplingArc; i++)
	{
		unsigned int weight = populationDensitiesSamplingBuffer[i] * distancesSamplingBuffer[i];

		if (weight > highestWeight)
		{
			highestWeight = weight;
			angleIncrement = (float)i;
			j = i;
		}
	}

	distance = distancesSamplingBuffer[j];
	goal = start + vml_rotate2D(vml_vec2(0.0f, (float)distance), startingAngle + vml_radians(angleIncrement - (float)configuration->halfSamplingArc));
}

//////////////////////////////////////////////////////////////////////////
void applyHighwayGoalDeviation(RoadAttributes& roadAttributes)
{
	if (configuration->maxHighwayGoalDeviation == 0)
	{
		return;
	}

	roadAttributes.angle += vml_radians((float)(rand() % configuration->halfMaxHighwayGoalDeviation) - (int)configuration->maxHighwayGoalDeviation);
}

//////////////////////////////////////////////////////////////////////////
void applyNaturalPatternRule(const vml_vec2& position, unsigned int goalDistance, int& delay, RoadAttributes& roadAttributes, RuleAttributes& ruleAttributes)
{
	roadAttributes.length = MathExtras::min(goalDistance, configuration->highwayLength);
	roadAttributes.angle = MathExtras::getOrientedAngle(vml_vec2(0.0f, 1.0f), ruleAttributes.goal - position);
	applyHighwayGoalDeviation(roadAttributes);
}

//////////////////////////////////////////////////////////////////////////
void applyRadialPatternRule(const vml_vec2& position, unsigned int goalDistance, int& delay, RoadAttributes& roadAttributes, RuleAttributes& ruleAttributes)
{
	// TODO:
}

//////////////////////////////////////////////////////////////////////////
void applyRasterPatternRule(const vml_vec2& position, unsigned int goalDistance, int& delay, RoadAttributes& roadAttributes, RuleAttributes& ruleAttributes)
{
	float angle = MathExtras::getOrientedAngle(vml_vec2(1.0f, 0.0f), ruleAttributes.goal - position);
	unsigned int horizontalDistance = (unsigned int)abs((float)goalDistance * cos(angle));
	unsigned int verticalDistance = (unsigned int)abs((float)goalDistance * sin(angle));
	bool canMoveHorizontally = horizontalDistance >= configuration->minRoadLength;
	bool canMoveVertically = verticalDistance >= configuration->minRoadLength;
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

	unsigned int length = (rand() % (configuration->highwayLength - configuration->minRoadLength)) + configuration->minRoadLength;

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