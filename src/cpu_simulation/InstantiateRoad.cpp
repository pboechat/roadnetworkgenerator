#include <InstantiateRoad.h>
#include <EvaluateBranch.h>
#include <EvaluateRoad.h>

#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>

#include <exception>

InstantiateRoad::InstantiateRoad(const Road& road) : road(road)
{
}

unsigned int InstantiateRoad::getCode()
{
	return 0;
}

void InstantiateRoad::execute(WorkQueuesManager<Procedure>& workQueuesManager, std::vector<Segment>& segments, ImageMap& populationDensityMap, ImageMap& waterBodiesMap)
{
	// p2

	// FIXME: checking invariants
	if (road.state != SUCCEED)
	{
		throw std::exception("road.state != SUCCEED");
	}

	glm::vec3 end(0.0f, road.roadAttributes.length, 0.0f);
	end = glm::rotate(glm::quat(glm::vec3(0, 0, road.roadAttributes.angle)), end);
	segments.push_back(Segment(road.roadAttributes.start, end));
	int delays[3];
	RoadAttributes roadAttributes[3];
	RuleAttributes ruleAttributes[3];
	evaluateGlobalGoals(delays, roadAttributes, ruleAttributes);
	workQueuesManager.addWorkItem(new EvaluateBranch(Branch(delays[0], roadAttributes[0], ruleAttributes[0])));
	workQueuesManager.addWorkItem(new EvaluateBranch(Branch(delays[1], roadAttributes[1], ruleAttributes[1])));
	workQueuesManager.addWorkItem(new EvaluateRoad(Road(delays[2], roadAttributes[2], ruleAttributes[2], UNASSIGNED)));
}

void InstantiateRoad::evaluateGlobalGoals(int* delays, RoadAttributes* roadAttributes, RuleAttributes* ruleAttributes)
{
	delays[0] = 2;
	delays[1] = 2;
	delays[2] = 1;

	roadAttributes[0].start = road.roadAttributes.start;
	roadAttributes[0].length = road.roadAttributes.length;
	roadAttributes[0].angle = -90.0f;

	roadAttributes[1].start = road.roadAttributes.start;
	roadAttributes[1].length = road.roadAttributes.length;
	roadAttributes[1].angle = 90.0f;

	roadAttributes[2].start = road.roadAttributes.start;
	roadAttributes[2].length = road.roadAttributes.length;

	// TODO:
}
