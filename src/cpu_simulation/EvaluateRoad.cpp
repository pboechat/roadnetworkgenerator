#include <EvaluateRoad.h>
#include <InstantiateRoad.h>

#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>

#include <exception>

EvaluateRoad::EvaluateRoad(const Road& road) : road(road)
{
}

unsigned int EvaluateRoad::getCode()
{
	return 2;
}

void EvaluateRoad::execute(WorkQueuesManager<Procedure>& workQueuesManager, RoadNetworkGraph::Graph& roadNetworkGraph, const Configuration& configuration)
{
	// p1, p3 and p6
	if (road.delay < 0 || road.state == FAILED)
	{
		return;
	}

	// p8
	if (road.state == UNASSIGNED)
	{
		evaluateLocalContraints(configuration, roadNetworkGraph);

		// FIXME: checking invariants
		if (road.state == UNASSIGNED)
		{
			throw std::exception("road.state == UNASSIGNED");
		}
	}

	if (road.state == FAILED)
	{
		workQueuesManager.addWorkItem(new EvaluateRoad(road));
	}

	else if (road.state == SUCCEED)
	{
		workQueuesManager.addWorkItem(new InstantiateRoad(road));
	}
}

void EvaluateRoad::evaluateLocalContraints(const Configuration& configuration, const RoadNetworkGraph::Graph& roadNetworkGraph)
{
	// remove streets that have exceeded max street branch depth
	if (!road.roadAttributes.highway && road.ruleAttributes.streetBranchDepth > configuration.maxStreetBranchDepth)
	{
		road.state = FAILED;
		return;
	}

	glm::vec3 position = roadNetworkGraph.getPosition(road.roadAttributes.source);

	// remove roads that cross world boundaries
	if (position.x < 0 || position.x > (float)configuration.worldWidth ||
		position.y < 0 || position.y > (float)configuration.worldHeight)
	{
		road.state = FAILED;
		return;
	}

	unsigned int angleIncrement = 0;

	unsigned int length = road.roadAttributes.length;
	while (length >= configuration.minRoadLength)
	{
		do
		{
			glm::vec3 direction = glm::normalize(glm::rotate(glm::quat(glm::vec3(0, 0, glm::radians(road.roadAttributes.angle + (float)angleIncrement))), glm::vec3(0.0f, 1.0f, 0.0f)));

			if (configuration.waterBodiesMap.castRay(position, direction, length, 0))
			{
				road.state = SUCCEED;
				break;
			}

			angleIncrement++;;
		}
		while (angleIncrement <= configuration.maxObstacleDeviationAngle);

		if (road.state == SUCCEED)
		{
			break;
		}

		length--;
	}

	road.roadAttributes.length = length;

	if (angleIncrement > configuration.maxObstacleDeviationAngle)
	{
		road.state = FAILED;
	}

	else
	{
		road.roadAttributes.angle += angleIncrement;
	}
}