#include <InstantiateRoad.h>
#include <EvaluateBranch.h>
#include <EvaluateRoad.h>

#include <glm/gtx/quaternion.hpp>

#include <exception>

InstantiateRoad::InstantiateRoad(const Road& road) : road(road)
{
}

unsigned int InstantiateRoad::getCode()
{
	return 0;
}

void InstantiateRoad::execute(WorkQueuesManager<Procedure>& workQueuesManager, std::vector<Segment>& segments, const Configuration& configuration)
{
	// p2

	// FIXME: checking invariants
	if (road.state != SUCCEED)
	{
		throw std::exception("road.state != SUCCEED");
	}

	glm::vec3 direction = glm::normalize(glm::rotate(glm::quat(glm::vec3(0, 0, glm::radians(road.roadAttributes.angle))), glm::vec3(0.0f, 1.0f, 0.0f)));
	glm::vec3 end = road.roadAttributes.start + (direction * (float)road.roadAttributes.length);
	segments.push_back(Segment(road.roadAttributes.start, end));
	int delays[3];
	RoadAttributes roadAttributes[3];
	RuleAttributes ruleAttributes[3];
	evaluateGlobalGoals(configuration, end, delays, roadAttributes, ruleAttributes);
	workQueuesManager.addWorkItem(new EvaluateBranch(Branch(delays[0], roadAttributes[0], ruleAttributes[0])));
	workQueuesManager.addWorkItem(new EvaluateBranch(Branch(delays[1], roadAttributes[1], ruleAttributes[1])));
	workQueuesManager.addWorkItem(new EvaluateRoad(Road(delays[2], roadAttributes[2], ruleAttributes[2], UNASSIGNED)));
}

void InstantiateRoad::evaluateGlobalGoals(const Configuration& configuration, const glm::vec3& roadEnd, int* delays, RoadAttributes* roadAttributes, RuleAttributes* ruleAttributes)
{
	if (road.roadAttributes.highway)
	{
		delays[0] = 2;
		delays[1] = 2;
		delays[2] = 1;
		roadAttributes[0].start = roadEnd;
		roadAttributes[0].length = configuration.streetLength;
		roadAttributes[0].angle = road.roadAttributes.angle - 90.0f;
		roadAttributes[0].highway = false;
		roadAttributes[1].start = roadEnd;
		roadAttributes[1].length = configuration.streetLength;
		roadAttributes[1].angle = road.roadAttributes.angle + 90.0f;
		roadAttributes[1].highway = false;
		roadAttributes[2].start = roadEnd;
		roadAttributes[2].length = configuration.highwayLength;
		roadAttributes[2].angle = findHighwayAngle(roadEnd, road.roadAttributes.angle, configuration.highwayLength, configuration);
		roadAttributes[2].highway = true;
	}

	else
	{
		delays[0] = 2;
		delays[1] = 2;
		delays[2] = 1;
		roadAttributes[0].start = roadEnd;
		roadAttributes[0].length = configuration.streetLength;
		roadAttributes[0].angle = road.roadAttributes.angle - 90.0f;
		roadAttributes[0].highway = false;
		roadAttributes[1].start = roadEnd;
		roadAttributes[1].length = configuration.streetLength;
		roadAttributes[1].angle = road.roadAttributes.angle + 90.0f;
		roadAttributes[1].highway = false;
		roadAttributes[2].start = roadEnd;
		roadAttributes[2].length = configuration.streetLength;
		roadAttributes[2].angle = road.roadAttributes.angle;
		roadAttributes[2].highway = false;
	}
}

float InstantiateRoad::findHighwayAngle(const glm::vec3& startingPoint, float startingAngle, int length, const Configuration& configuration) const
{
	char highestDensity = -128;
	int currentAngle = -(configuration.samplingArc + 1) / 2;
	int angle = 0;

	for (int i = 0; i < configuration.samplingArc; i++, currentAngle++)
	{
		glm::vec3 direction = glm::normalize(glm::rotate(glm::quat(glm::vec3(0.0f, 0.0f, glm::radians(startingAngle + (float)currentAngle))), glm::vec3(0.0f, 1.0f, 0.0f)));
		glm::vec3 point = startingPoint + (direction * (float)length);
		char density = configuration.populationDensityMap.sample(point);

		if (density > highestDensity)
		{
			angle = currentAngle;
			highestDensity = density;
		}
	}

	return startingAngle + angle;
}
