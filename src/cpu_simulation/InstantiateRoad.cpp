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
		delays[0] = 5;
		delays[1] = 5;
		delays[2] = 1;
		roadAttributes[0].start = roadEnd;
		roadAttributes[0].length = configuration.streetLength;
		roadAttributes[0].width = configuration.streetWidth;
		roadAttributes[0].angle = road.roadAttributes.angle - 90.0f;
		roadAttributes[0].highway = false;
		roadAttributes[1].start = roadEnd;
		roadAttributes[1].length = configuration.streetLength;
		roadAttributes[1].width = configuration.streetWidth;
		roadAttributes[1].angle = road.roadAttributes.angle + 90.0f;
		roadAttributes[1].highway = false;
		roadAttributes[2].start = roadEnd;
		roadAttributes[2].length = configuration.highwayLength;
		roadAttributes[2].width = configuration.highwayWidth;
		roadAttributes[2].angle = road.roadAttributes.angle;
		roadAttributes[2].highway = true;
		adjustHighwayAttributes(roadAttributes[2], configuration);
	}

	else
	{
		delays[0] = 5;
		delays[1] = 5;
		delays[2] = 5;
		roadAttributes[0].start = roadEnd;
		roadAttributes[0].length = configuration.streetLength;
		roadAttributes[0].width = configuration.streetWidth;
		roadAttributes[0].angle = road.roadAttributes.angle - 90.0f;
		roadAttributes[0].highway = false;
		roadAttributes[1].start = roadEnd;
		roadAttributes[1].length = configuration.streetLength;
		roadAttributes[1].width = configuration.streetWidth;
		roadAttributes[1].angle = road.roadAttributes.angle + 90.0f;
		roadAttributes[1].highway = false;
		roadAttributes[2].start = roadEnd;
		roadAttributes[2].length = configuration.streetLength;
		roadAttributes[2].width = configuration.streetWidth;
		roadAttributes[2].angle = road.roadAttributes.angle;
		roadAttributes[2].highway = false;
	}
}

void InstantiateRoad::adjustHighwayAttributes(RoadAttributes& highwayAttributes, const Configuration& configuration) const
{
	unsigned char highestDensity = 0;
	int currentAngleStep = -(configuration.samplingArc + 1) / 2;
	int angleIncrement = 0;
	int newLength = highwayAttributes.length;

	for (int i = 0; i < configuration.samplingArc; i++, currentAngleStep++)
	{
		glm::vec3 direction = glm::normalize(glm::rotate(glm::quat(glm::vec3(0.0f, 0.0f, glm::radians(highwayAttributes.angle + (float)currentAngleStep))), glm::vec3(0.0f, 1.0f, 0.0f)));
		glm::vec3 hit;
		unsigned char highestDensity_ray;
		int lengthStep;
		configuration.populationDensityMap.scan(highwayAttributes.start, direction, highwayAttributes.length, configuration.minHighwayLength, highestDensity_ray, lengthStep);

		if (highestDensity_ray > highestDensity)
		{
			angleIncrement = currentAngleStep;
			newLength = lengthStep;
			highestDensity = highestDensity_ray;
		}
	}

	highwayAttributes.angle += (float)angleIncrement;
	highwayAttributes.length = newLength;
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
