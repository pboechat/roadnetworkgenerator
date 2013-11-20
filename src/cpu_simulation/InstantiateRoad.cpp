#include <InstantiateRoad.h>
#include <EvaluateBranch.h>
#include <EvaluateRoad.h>
#include <Circle.h>

#include <glm/gtx/quaternion.hpp>

#include <exception>

InstantiateRoad::InstantiateRoad(const Road& road) : road(road)
{
}

unsigned int InstantiateRoad::getCode()
{
	return 0;
}

void InstantiateRoad::execute(WorkQueuesManager<Procedure>& workQueuesManager, QuadTree& quadtree, const Configuration& configuration)
{
	// p2

	// FIXME: checking invariants
	if (road.state != SUCCEED)
	{
		throw std::exception("road.state != SUCCEED");
	}

	glm::vec3 start = road.roadAttributes.start;
	glm::vec3 direction = glm::normalize(glm::rotate(glm::quat(glm::vec3(0, 0, glm::radians(road.roadAttributes.angle))), glm::vec3(0.0f, 1.0f, 0.0f)));
	glm::vec3 end = start + (direction * (float)road.roadAttributes.length);
	glm::vec3 newStart = snap(start, configuration, quadtree);
	glm::vec3 newEnd = snap(end, configuration, quadtree);
	glm::vec4 color1 = (newStart != start) ? configuration.snapColor : (road.roadAttributes.highway) ? configuration.highwayColor : configuration.streetColor;
	glm::vec4 color2 = (newEnd != end) ? configuration.snapColor : (road.roadAttributes.highway) ? configuration.highwayColor : configuration.streetColor;
	quadtree.insert(Line(newStart, newEnd, road.roadAttributes.width, color1, color2));
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
		bool doPureHighwayBranch = (road.ruleAttributes.pureHighwayBranchingDistance == configuration.minPureHighwayBranchingDistance);
		bool doRegularBranch = (road.ruleAttributes.highwayBranchingDistance == configuration.minHighwayBranchingDistance);
		delays[0] = (doPureHighwayBranch) ? 0 : (doRegularBranch) ? configuration.highwayBranchingDelay : -1;
		delays[1] = (doPureHighwayBranch) ? 0 : (doRegularBranch) ? configuration.highwayBranchingDelay : -1;
		delays[2] = 0;
		// new street/highway branch left
		roadAttributes[0].start = roadEnd;
		roadAttributes[0].length = (doPureHighwayBranch) ? configuration.highwayLength : configuration.streetLength;
		roadAttributes[0].width = (doPureHighwayBranch) ? configuration.highwayWidth : configuration.streetWidth;
		roadAttributes[0].angle = road.roadAttributes.angle - 90.0f;
		roadAttributes[0].highway = doPureHighwayBranch;
		// new street/highway branch right
		roadAttributes[1].start = roadEnd;
		roadAttributes[1].length = (doPureHighwayBranch) ? configuration.highwayLength : configuration.streetLength;
		roadAttributes[1].width = (doPureHighwayBranch) ? configuration.highwayWidth : configuration.streetWidth;
		roadAttributes[1].angle = road.roadAttributes.angle + 90.0f;
		roadAttributes[1].highway = doPureHighwayBranch;
		// main highway continuity
		roadAttributes[2].start = roadEnd;
		roadAttributes[2].length = configuration.highwayLength;
		roadAttributes[2].width = configuration.highwayWidth;
		roadAttributes[2].angle = road.roadAttributes.angle;
		roadAttributes[2].highway = true;
		ruleAttributes[2].highwayBranchingDistance = (doRegularBranch) ? 0 : road.ruleAttributes.highwayBranchingDistance + 1;
		ruleAttributes[2].pureHighwayBranchingDistance = (doPureHighwayBranch) ? 0 : road.ruleAttributes.pureHighwayBranchingDistance + 1;

		if (doPureHighwayBranch)
		{
			followHighestPopulationDensity(roadAttributes[0], ruleAttributes[0], configuration);
			followHighestPopulationDensity(roadAttributes[1], ruleAttributes[0], configuration);
		}

		ruleAttributes[2].highwayGoalDistance = road.ruleAttributes.highwayGoalDistance - road.roadAttributes.length;

		if (ruleAttributes[2].highwayGoalDistance <= 0)
		{
			followHighestPopulationDensity(roadAttributes[2], ruleAttributes[2], configuration);
		}
	}

	else
	{
		delays[0] = configuration.streetBranchingDelay;
		delays[1] = configuration.streetBranchingDelay;
		delays[2] = 0;
		unsigned int newStreetDepth = road.ruleAttributes.streetBranchDepth + 1;
		// street branch left
		roadAttributes[0].start = roadEnd;
		roadAttributes[0].length = configuration.streetLength;
		roadAttributes[0].width = configuration.streetWidth;
		roadAttributes[0].angle = road.roadAttributes.angle - 90.0f;
		roadAttributes[0].highway = false;
		ruleAttributes[0].streetBranchDepth = newStreetDepth;
		// street branch right
		roadAttributes[1].start = roadEnd;
		roadAttributes[1].length = configuration.streetLength;
		roadAttributes[1].width = configuration.streetWidth;
		roadAttributes[1].angle = road.roadAttributes.angle + 90.0f;
		roadAttributes[1].highway = false;
		ruleAttributes[1].streetBranchDepth = newStreetDepth;
		// street
		roadAttributes[2].start = roadEnd;
		roadAttributes[2].length = configuration.streetLength;
		roadAttributes[2].width = configuration.streetWidth;
		roadAttributes[2].angle = road.roadAttributes.angle;
		roadAttributes[2].highway = false;
		ruleAttributes[2].streetBranchDepth = newStreetDepth;
	}
}

void InstantiateRoad::followHighestPopulationDensity(RoadAttributes& roadAttributes, RuleAttributes& ruleAttributes, const Configuration& configuration) const
{
	int halfSamplingArc = ((int)configuration.samplingArc + 1) / 2;
	int currentAngleStep = -halfSamplingArc;
	unsigned char* populationDensities = new unsigned char[configuration.samplingArc];
	int* distances = new int[configuration.samplingArc];

	for (unsigned int i = 0; i < configuration.samplingArc; i++, currentAngleStep++)
	{
		glm::vec3 direction = glm::normalize(glm::rotate(glm::quat(glm::vec3(0.0f, 0.0f, glm::radians(roadAttributes.angle + (float)currentAngleStep))), glm::vec3(0.0f, 1.0f, 0.0f)));
		configuration.populationDensityMap.scan(roadAttributes.start, direction, configuration.minSamplingRayLength, configuration.maxSamplingRayLength, populationDensities[i], distances[i]);
	}

	unsigned int highestWeight = 0;
	unsigned int j = 0;

	for (unsigned int i = 0; i < configuration.samplingArc; i++)
	{
		unsigned int weight = populationDensities[i] * distances[i];

		if (weight > highestWeight)
		{
			highestWeight = weight;
			j = i;
		}
	}

	roadAttributes.angle = roadAttributes.angle + (j - halfSamplingArc);
	ruleAttributes.highwayGoalDistance = distances[j];
	delete[] populationDensities;
	delete[] distances;
}

glm::vec3 InstantiateRoad::snap(const glm::vec3& point, const Configuration& configuration, QuadTree& quadtree) const
{
	std::vector<Line> neighbours;
	quadtree.query(Circle(point, (float)configuration.quadtreeQueryRadius), neighbours);
	// FIXME:
	float minDistance = 100000.0f;
	glm::vec3 closestPoint = point;

	// FIXME: horrible code!!!
	for (unsigned int i = 0; i < neighbours.size(); i++)
	{
		glm::vec3 snapPoint = neighbours[i].start;
		float distance = glm::distance(snapPoint, point);

		if (distance < configuration.quadtreeQueryRadius && distance < minDistance)
		{
			minDistance = distance;
			closestPoint = snapPoint;
		}

		snapPoint = neighbours[i].end;
		distance = glm::distance(snapPoint, point);

		if (distance < configuration.quadtreeQueryRadius && distance < minDistance)
		{
			minDistance = distance;
			closestPoint = snapPoint;
		}
	}

	return closestPoint;
}
