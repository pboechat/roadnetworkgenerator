#include <InstantiateRoad.h>
#include <EvaluateBranch.h>
#include <EvaluateRoad.h>
#include <Circle.h>
#include <Pattern.h>
#include <MathExtras.h>

#include <glm/gtx/quaternion.hpp>

#include <sstream>
#include <random>
#include <exception>

unsigned char* InstantiateRoad::populationDensities = 0;
unsigned int* InstantiateRoad::distances = 0;

InstantiateRoad::InstantiateRoad()
{
}

InstantiateRoad::InstantiateRoad(const Road& road) : road(road)
{
}

unsigned int InstantiateRoad::getCode() const
{
	return INSTANTIATE_ROAD_CODE;
}

void InstantiateRoad::execute(WorkQueuesManager& manager, RoadNetworkGraph::Graph& graph, const Configuration& configuration)
{
	// p2

	// FIXME: checking invariants
	if (road.state != SUCCEED)
	{
		throw std::exception("road.state != SUCCEED");
	}

	glm::vec3 direction = glm::rotate(glm::quat(glm::vec3(0, 0, road.roadAttributes.angle)), glm::vec3(0.0f, road.roadAttributes.length, 0.0f));

	RoadNetworkGraph::VertexIndex newSource;
	glm::vec3 position;
	bool interrupted = graph.addRoad(road.roadAttributes.source, direction, newSource, position, road.roadAttributes.highway);

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
		evaluateGlobalGoals(configuration, newSource, position, delays, roadAttributes, ruleAttributes);
	}

	manager.addWorkItem(EvaluateBranch(Branch(delays[0], roadAttributes[0], ruleAttributes[0])));
	manager.addWorkItem(EvaluateBranch(Branch(delays[1], roadAttributes[1], ruleAttributes[1])));
	manager.addWorkItem(EvaluateRoad(Road(delays[2], roadAttributes[2], ruleAttributes[2], UNASSIGNED)));
}

void InstantiateRoad::evaluateGlobalGoals(const Configuration& configuration, RoadNetworkGraph::VertexIndex source, const glm::vec3& position, int* delays, RoadAttributes* roadAttributes, RuleAttributes* ruleAttributes)
{
	if (road.roadAttributes.highway)
	{
		bool doPureHighwayBranch = (road.ruleAttributes.pureHighwayBranchingDistance == configuration.minPureHighwayBranchingDistance);
		bool doRegularBranch = (road.ruleAttributes.highwayBranchingDistance == configuration.minHighwayBranchingDistance);

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
			findHighestPopulationDensity(configuration, position, road.roadAttributes.angle, ruleAttributes[2].goal, goalDistance);
			ruleAttributes[2].hasGoal = true;
		}

		else
		{
			goalDistance = (unsigned int)glm::distance(position, road.ruleAttributes.goal);
		}
		
		if (goalDistance <= configuration.goalDistanceThreshold)
		{
			delays[2] = -1; // remove highway
		}

		else
		{
			Pattern pattern = findUnderlyingPattern(configuration, position);
			if (pattern == NATURAL_PATTERN)
			{
				applyNaturalPatternRule(configuration, position, goalDistance, delays[2], roadAttributes[2], ruleAttributes[2]);
			}

			else if (pattern == RADIAL_PATTERN)
			{
				applyRadialPatternRule(configuration, position, goalDistance, delays[2], roadAttributes[2], ruleAttributes[2]);
			}

			else if (pattern == RASTER_PATTERN)
			{
				applyRasterPatternRule(configuration, position, goalDistance, delays[2], roadAttributes[2], ruleAttributes[2]);
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
			roadAttributes[0].length = configuration.highwayLength;
			roadAttributes[0].angle = roadAttributes[2].angle - MathExtras::HALF_PI;
			roadAttributes[0].highway = true;

			//applyHighwayGoalDeviation(configuration, roadAttributes[0]);

			// new highway branch right
			delays[1] = 0;
			roadAttributes[1].source = source;
			roadAttributes[1].length = configuration.highwayLength;
			roadAttributes[1].angle = roadAttributes[2].angle + MathExtras::HALF_PI;
			roadAttributes[1].highway = true;

			//applyHighwayGoalDeviation(configuration, roadAttributes[1]);
		}

		else
		{
			// new street branch left
			delays[0] = (doRegularBranch) ? configuration.highwayBranchingDelay : -1;
			roadAttributes[0].source = source;
			roadAttributes[0].length = configuration.streetLength;
			roadAttributes[0].angle = roadAttributes[2].angle - MathExtras::HALF_PI;
			roadAttributes[0].highway = false;

			// new street branch right
			delays[1] = (doRegularBranch) ? configuration.highwayBranchingDelay : -1;
			roadAttributes[1].source = source;
			roadAttributes[1].length = configuration.streetLength;
			roadAttributes[1].angle = roadAttributes[2].angle + MathExtras::HALF_PI;
			roadAttributes[1].highway = false;
		}
	}

	else
	{
		unsigned int newStreetDepth = road.ruleAttributes.streetBranchDepth + 1;

		// street branch left
		delays[0] = configuration.streetBranchingDelay;
		roadAttributes[0].source = source;
		roadAttributes[0].length = configuration.streetLength;
		roadAttributes[0].angle = road.roadAttributes.angle - MathExtras::HALF_PI;
		roadAttributes[0].highway = false;
		ruleAttributes[0].streetBranchDepth = newStreetDepth;

		// street branch right
		delays[1] = configuration.streetBranchingDelay;
		roadAttributes[1].source = source;
		roadAttributes[1].length = configuration.streetLength;
		roadAttributes[1].angle = road.roadAttributes.angle + MathExtras::HALF_PI;
		roadAttributes[1].highway = false;
		ruleAttributes[1].streetBranchDepth = newStreetDepth;

		// street continuation
		delays[2] = 0;
		roadAttributes[2].source = source;
		roadAttributes[2].length = configuration.streetLength;
		roadAttributes[2].angle = road.roadAttributes.angle;
		roadAttributes[2].highway = false;
		ruleAttributes[2].streetBranchDepth = newStreetDepth;
	}
}

Pattern InstantiateRoad::findUnderlyingPattern(const Configuration& configuration, const glm::vec3& position) const
{
	unsigned char naturalPattern = 0;
	if (configuration.naturalPatternMap != 0)
	{
		naturalPattern = configuration.naturalPatternMap->sample(position);
	}

	unsigned char radialPattern = 0;
	if (configuration.radialPatternMap != 0)
	{
		radialPattern = configuration.radialPatternMap->sample(position);
	}

	unsigned char rasterPattern = 0;
	if (configuration.rasterPatternMap != 0)
	{
		rasterPattern = configuration.rasterPatternMap->sample(position);
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

void InstantiateRoad::findHighestPopulationDensity(const Configuration& configuration, const glm::vec3& start, float startingAngle, glm::vec3& goal, unsigned int& distance) const
{
	int currentAngleStep = -configuration.halfSamplingArc;

	// FIXME: checking invariants
	if (configuration.populationDensityMap == 0)
	{
		throw std::exception("configuration.populationDensityMap == 0");
	}

	for (unsigned int i = 0; i < configuration.samplingArc; i++, currentAngleStep++)
	{
		glm::vec3 direction = glm::normalize(glm::rotate(glm::quat(glm::vec3(0.0f, 0.0f, startingAngle + glm::radians((float)currentAngleStep))), glm::vec3(0.0f, 1.0f, 0.0f)));
		unsigned char populationDensity;
		int distance;
		configuration.populationDensityMap->scan(start, direction, configuration.minSamplingRayLength, configuration.maxSamplingRayLength, populationDensity, distance);
		populationDensities[i] = populationDensity;
		distances[i] = distance;
	}

	unsigned int highestWeight = 0;
	unsigned int j = 0;
	float angleIncrement = 0.0f;

	for (unsigned int i = 0; i < configuration.samplingArc; i++)
	{
		unsigned int weight = populationDensities[i] * distances[i];

		if (weight > highestWeight)
		{
			highestWeight = weight;
			angleIncrement = (float)i;
			j = i;
		}
	}

	float angle = startingAngle + glm::radians(angleIncrement - (float)configuration.halfSamplingArc);
	distance = distances[j];
	goal = start + glm::rotate(glm::quat(glm::vec3(0.0f, 0.0f, angle)), glm::vec3(0.0f, (float)distance, 0.0f));
}

void InstantiateRoad::applyHighwayGoalDeviation(const Configuration& configuration, RoadAttributes& roadAttributes) const
{
	if (configuration.maxHighwayGoalDeviation == 0)
	{
		return;
	}

	roadAttributes.angle += glm::radians((float)(rand() % configuration.halfMaxHighwayGoalDeviation) - (int)configuration.maxHighwayGoalDeviation);
}

void InstantiateRoad::applyNaturalPatternRule(const Configuration& configuration, const glm::vec3& position, unsigned int goalDistance, int& delay, RoadAttributes& roadAttributes, RuleAttributes& ruleAttributes) const
{
	roadAttributes.length = MathExtras::min(goalDistance, configuration.highwayLength);
	roadAttributes.angle = MathExtras::getOrientedAngle(glm::vec3(0.0f, 1.0f, 0.0f), ruleAttributes.goal - position);
	applyHighwayGoalDeviation(configuration, roadAttributes);
}

void InstantiateRoad::applyRadialPatternRule(const Configuration& configuration, const glm::vec3& position, unsigned int goalDistance, int& delay, RoadAttributes& roadAttributes, RuleAttributes& ruleAttributes) const
{
	// TODO:
}

void InstantiateRoad::applyRasterPatternRule(const Configuration& configuration, const glm::vec3& position, unsigned int goalDistance, int& delay, RoadAttributes& roadAttributes, RuleAttributes& ruleAttributes) const
{
	float angle = MathExtras::getOrientedAngle(glm::vec3(1.0f, 0.0f, 0.0f), ruleAttributes.goal - position);

	unsigned int horizontalDistance = (unsigned int)glm::abs((float)goalDistance * glm::cos(angle));
	unsigned int verticalDistance = (unsigned int)glm::abs((float)goalDistance * glm::sin(angle));

	bool canMoveHorizontally = horizontalDistance >= configuration.minRoadLength;
	bool canMoveVertically = verticalDistance >= configuration.minRoadLength;

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

	unsigned int length = (rand() % (configuration.highwayLength - configuration.minRoadLength)) + configuration.minRoadLength;
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

void InstantiateRoad::initialize(const Configuration& configuration)
{
	populationDensities = new unsigned char[configuration.samplingArc];
	distances = new unsigned int[configuration.samplingArc];
}

void InstantiateRoad::dispose()
{
	if (populationDensities != 0)
	{
		delete[] populationDensities;
	}

	if (distances != 0)
	{
		delete[] distances;
	}
}