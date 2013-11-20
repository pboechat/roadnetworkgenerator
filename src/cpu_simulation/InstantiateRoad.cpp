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

		// street/highway branch left
		roadAttributes[0].start = roadEnd;
		roadAttributes[0].length = (doPureHighwayBranch) ? configuration.highwayLength : configuration.streetLength;
		roadAttributes[0].width = (doPureHighwayBranch) ? configuration.highwayWidth : configuration.streetWidth;
		roadAttributes[0].angle = road.roadAttributes.angle - 90.0f;
		roadAttributes[0].highway = doPureHighwayBranch;
		// street/highway branch right
		roadAttributes[1].start = roadEnd;
		roadAttributes[1].length = (doPureHighwayBranch) ? configuration.highwayLength : configuration.streetLength;
		roadAttributes[1].width = (doPureHighwayBranch) ? configuration.highwayWidth : configuration.streetWidth;
		roadAttributes[1].angle = road.roadAttributes.angle + 90.0f;
		roadAttributes[1].highway = doPureHighwayBranch;
		// highway
		roadAttributes[2].start = roadEnd;
		roadAttributes[2].length = configuration.highwayLength;
		roadAttributes[2].width = configuration.highwayWidth;
		roadAttributes[2].angle = road.roadAttributes.angle;
		roadAttributes[2].highway = true;
		ruleAttributes[2].highwayBranchingDistance = (doRegularBranch) ? 0 : road.ruleAttributes.highwayBranchingDistance + 1;
		ruleAttributes[2].pureHighwayBranchingDistance = (doPureHighwayBranch) ? 0 : road.ruleAttributes.pureHighwayBranchingDistance + 1;
		if (doPureHighwayBranch)
		{
			adjustHighwayAttributes(roadAttributes[0], configuration);
			adjustHighwayAttributes(roadAttributes[1], configuration);
		}
		adjustHighwayAttributes(roadAttributes[2], configuration);
	}

	else
	{
		delays[0] = configuration.streetBranchingDelay;
		delays[1] = configuration.streetBranchingDelay;
		delays[2] = 0;
		int newDepth = road.ruleAttributes.streetBranchDepth + 1;
		// street branch left
		roadAttributes[0].start = roadEnd;
		roadAttributes[0].length = configuration.streetLength;
		roadAttributes[0].width = configuration.streetWidth;
		roadAttributes[0].angle = road.roadAttributes.angle - 90.0f;
		roadAttributes[0].highway = false;
		ruleAttributes[0].streetBranchDepth = newDepth;
		// street branch right
		roadAttributes[1].start = roadEnd;
		roadAttributes[1].length = configuration.streetLength;
		roadAttributes[1].width = configuration.streetWidth;
		roadAttributes[1].angle = road.roadAttributes.angle + 90.0f;
		roadAttributes[1].highway = false;
		ruleAttributes[1].streetBranchDepth = newDepth;
		// street
		roadAttributes[2].start = roadEnd;
		roadAttributes[2].length = configuration.streetLength;
		roadAttributes[2].width = configuration.streetWidth;
		roadAttributes[2].angle = road.roadAttributes.angle;
		roadAttributes[2].highway = false;
		ruleAttributes[2].streetBranchDepth = newDepth;
	}
}

void InstantiateRoad::adjustHighwayAttributes(RoadAttributes& highwayAttributes, const Configuration& configuration) const
{
	int halfSamplingArc = -(configuration.samplingArc + 1) / 2;
	int currentAngleStep = halfSamplingArc;
	unsigned char* densities = new unsigned char[configuration.samplingArc];
	int* lengths = new int[configuration.samplingArc];
	for (int i = 0; i < configuration.samplingArc; i++, currentAngleStep++)
	{
		glm::vec3 direction = glm::normalize(glm::rotate(glm::quat(glm::vec3(0.0f, 0.0f, glm::radians(highwayAttributes.angle + (float)currentAngleStep))), glm::vec3(0.0f, 1.0f, 0.0f)));
		configuration.populationDensityMap.scan(highwayAttributes.start, direction, highwayAttributes.length, configuration.minHighwayLength, densities[i], lengths[i]);
	}

	int highestScore = 0;
	int j = 0;
	for (int i = 0; i < configuration.samplingArc; i++)
	{
		int score = densities[i] * lengths[i];
		if (score > highestScore)
		{
			highestScore = score;
			j = i;
		}
	}

	highwayAttributes.angle += (halfSamplingArc + j);
	highwayAttributes.length = lengths[j];

	delete[] densities;
	delete[] lengths;
}

glm::vec3 InstantiateRoad::snap(const glm::vec3& point, const Configuration& configuration, QuadTree& quadtree) const
{
	std::vector<Line> neighbours;
	quadtree.query(Circle(point, (float)configuration.quadtreeQueryRadius), neighbours);
	// FIXME:
	float minDistance = 100000.0f;
	glm::vec3 closestPoint = point;

	for (unsigned int i = 0; i < neighbours.size(); i++)
	{
		glm::vec3 snapPoint = neighbours[i].snap(point);
		float distance = glm::distance(snapPoint, point);

		if (distance < minDistance)
		{
			minDistance = distance;
			closestPoint = snapPoint;
		}
	}

	return closestPoint;
}
