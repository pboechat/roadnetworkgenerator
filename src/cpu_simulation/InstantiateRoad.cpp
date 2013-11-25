#include <InstantiateRoad.h>
#include <EvaluateBranch.h>
#include <EvaluateRoad.h>
#include <Circle.h>
#include <random>

#include <glm/gtx/quaternion.hpp>

#include <exception>

unsigned char* InstantiateRoad::populationDensities = 0;
unsigned int* InstantiateRoad::distances = 0;

InstantiateRoad::InstantiateRoad(const Road& road) : road(road)
{
}

unsigned int InstantiateRoad::getCode()
{
	return 0;
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

void InstantiateRoad::execute(WorkQueuesManager<Procedure>& workQueuesManager, RoadNetworkGraph::Graph& roadNetworkGraph, const Configuration& configuration)
{
	// p2

	// FIXME: checking invariants
	if (road.state != SUCCEED)
	{
		throw std::exception("road.state != SUCCEED");
	}

	glm::vec3 direction = glm::rotate(glm::quat(glm::vec3(0, 0, glm::radians(road.roadAttributes.angle))), glm::vec3(0.0f, (float)road.roadAttributes.length, 0.0f));

	RoadNetworkGraph::VertexIndex newSource;
	glm::vec3 position;
	float length;
	bool interrupted = roadNetworkGraph.addRoad(road.roadAttributes.source, direction, newSource, position, length, road.roadAttributes.highway);

	// DEBUG:
	if (aroundPoint(position, 115, 775, 10))
	{
		int a = 0;
	}

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
		evaluateGlobalGoals(configuration, newSource, position, length, delays, roadAttributes, ruleAttributes);
	}

	workQueuesManager.addWorkItem(new EvaluateBranch(Branch(delays[0], roadAttributes[0], ruleAttributes[0])));
	workQueuesManager.addWorkItem(new EvaluateBranch(Branch(delays[1], roadAttributes[1], ruleAttributes[1])));
	workQueuesManager.addWorkItem(new EvaluateRoad(Road(delays[2], roadAttributes[2], ruleAttributes[2], UNASSIGNED)));
}

void InstantiateRoad::evaluateGlobalGoals(const Configuration& configuration, RoadNetworkGraph::VertexIndex source, const glm::vec3& position, float length, int* delays, RoadAttributes* roadAttributes, RuleAttributes* ruleAttributes)
{
	if (road.roadAttributes.highway)
	{
		bool doPureHighwayBranch = (road.ruleAttributes.pureHighwayBranchingDistance == configuration.minPureHighwayBranchingDistance);
		bool doRegularBranch = (road.ruleAttributes.highwayBranchingDistance == configuration.minHighwayBranchingDistance);

		// highway continuation
		delays[2] = 0;
		roadAttributes[2].source = source;
		roadAttributes[2].length = configuration.highwayLength;
		roadAttributes[2].angle = road.roadAttributes.angle;
		roadAttributes[2].highway = true;
		ruleAttributes[2].hasGoal = road.ruleAttributes.hasGoal;

		ruleAttributes[2].highwayBranchingDistance = (doRegularBranch) ? 0 : road.ruleAttributes.highwayBranchingDistance + 1;
		ruleAttributes[2].pureHighwayBranchingDistance = (doPureHighwayBranch) ? 0 : road.ruleAttributes.pureHighwayBranchingDistance + 1;

		if (!ruleAttributes[2].hasGoal)
		{
			followHighestPopulationDensity(configuration, position, roadAttributes[2], ruleAttributes[2]);
		}
		else
		{
			ruleAttributes[2].goalDistance = road.ruleAttributes.goalDistance - length;
			if (ruleAttributes[2].goalDistance <= 0)
			{
				delays[2] = -1; // remove highway
				//return;
			}
			else
			{
				applyAngleDeviation(configuration, roadAttributes[2]);
			}
		}

		if (doPureHighwayBranch)
		{
			// new highway branch left
			delays[0] = 0;
			roadAttributes[0].source = source;
			roadAttributes[0].length = configuration.highwayLength;
			roadAttributes[0].angle = roadAttributes[2].angle - 90.0f;
			roadAttributes[0].highway = true;

			applyAngleDeviation(configuration, roadAttributes[0]);

			// new highway branch right
			delays[1] = 0;
			roadAttributes[1].source = source;
			roadAttributes[1].length = configuration.highwayLength;
			roadAttributes[1].angle = roadAttributes[2].angle + 90.0f;
			roadAttributes[1].highway = true;

			applyAngleDeviation(configuration, roadAttributes[1]);
		}

		else
		{
			// new street branch left
			delays[0] = (doRegularBranch) ? configuration.highwayBranchingDelay : -1;
			roadAttributes[0].source = source;
			roadAttributes[0].length = configuration.streetLength;
			roadAttributes[0].angle = roadAttributes[2].angle - 90.0f;
			roadAttributes[0].highway = false;

			// new street branch right
			delays[1] = (doRegularBranch) ? configuration.highwayBranchingDelay : -1;
			roadAttributes[1].source = source;
			roadAttributes[1].length = configuration.streetLength;
			roadAttributes[1].angle = roadAttributes[2].angle + 90.0f;
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
		roadAttributes[0].angle = road.roadAttributes.angle - 90.0f;
		roadAttributes[0].highway = false;
		ruleAttributes[0].streetBranchDepth = newStreetDepth;

		// street branch right
		delays[1] = configuration.streetBranchingDelay;
		roadAttributes[1].source = source;
		roadAttributes[1].length = configuration.streetLength;
		roadAttributes[1].angle = road.roadAttributes.angle + 90.0f;
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

void InstantiateRoad::followHighestPopulationDensity(const Configuration& configuration, const glm::vec3& start, RoadAttributes& roadAttributes, RuleAttributes& ruleAttributes) const
{
	unsigned int halfSamplingArc = (configuration.samplingArc + 1) / 2;
	int currentAngleStep = -(int)halfSamplingArc;

	for (unsigned int i = 0; i < configuration.samplingArc; i++, currentAngleStep++)
	{
		glm::vec3 direction = glm::normalize(glm::rotate(glm::quat(glm::vec3(0.0f, 0.0f, glm::radians(roadAttributes.angle + (float)currentAngleStep))), glm::vec3(0.0f, 1.0f, 0.0f)));
		unsigned char populationDensity;
		int distance;
		configuration.populationDensityMap.scan(start, direction, configuration.minSamplingRayLength, configuration.maxSamplingRayLength, populationDensity, distance);
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

	roadAttributes.angle += (angleIncrement - halfSamplingArc);
	ruleAttributes.goalDistance = (float)distances[j];
	ruleAttributes.hasGoal = true;
}

void InstantiateRoad::applyAngleDeviation(const Configuration& configuration, RoadAttributes& roadAttributes) const
{
	int halfMaxDeviation = configuration.maxHighwayGoalDeviation / 2;
	roadAttributes.angle += ((rand() % halfMaxDeviation) - halfMaxDeviation);
}