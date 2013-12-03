#include <EvaluateRoad.h>
#include <InstantiateRoad.h>

#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>

#include <exception>

EvaluateRoad::EvaluateRoad()
{
}

EvaluateRoad::EvaluateRoad(const Road& road) : road(road)
{
}

unsigned int EvaluateRoad::getCode() const
{
	return EVALUATE_ROAD_CODE;
}

void EvaluateRoad::execute(WorkQueuesManager& manager, RoadNetworkGraph::Graph& graph, const Configuration& configuration)
{
	// p1, p3 and p6
	if (road.delay < 0 || road.state == FAILED)
	{
		return;
	}

	// p8
	if (road.state == UNASSIGNED)
	{
		evaluateLocalContraints(configuration, graph);

		// FIXME: checking invariants
		if (road.state == UNASSIGNED)
		{
			throw std::exception("road.state == UNASSIGNED");
		}
	}

	if (road.state == FAILED)
	{
		manager.addWorkItem(EvaluateRoad(road));
	}

	else if (road.state == SUCCEED)
	{
		manager.addWorkItem(InstantiateRoad(road));
	}
}

void EvaluateRoad::evaluateLocalContraints(const Configuration& configuration, const RoadNetworkGraph::Graph& graph)
{
	// remove streets that have exceeded max street branch depth
	if (!road.roadAttributes.highway && road.ruleAttributes.streetBranchDepth > configuration.maxStreetBranchDepth)
	{
		road.state = FAILED;
		return;
	}

	glm::vec3 position = graph.getPosition(road.roadAttributes.source);

	// remove roads that cross world boundaries
	if (position.x < 0 || position.x > (float)configuration.worldWidth ||
		position.y < 0 || position.y > (float)configuration.worldHeight)
	{
		road.state = FAILED;
		return;
	}

	if (!evaluateWaterBodies(configuration, position))
	{
		return;
	}

	evaluateBlockades(configuration, position);
}

bool EvaluateRoad::evaluateWaterBodies(const Configuration &configuration, const glm::vec3& position)
{
	unsigned int angleIncrement = 0;
	unsigned int length = road.roadAttributes.length;
	while (length >= configuration.minRoadLength)
	{
		do
		{
			glm::vec3 direction = glm::normalize(glm::rotate(glm::quat(glm::vec3(0, 0, road.roadAttributes.angle + glm::radians((float)angleIncrement))), glm::vec3(0.0f, 1.0f, 0.0f)));

			// FIXME: checking invariants
			if (configuration.waterBodiesMap == 0)
			{
				throw std::exception("configuration.waterBodiesMap == 0");
			}

			if (configuration.waterBodiesMap->castRay(position, direction, length, 0))
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
		return false;
	}

	else
	{
		road.roadAttributes.angle += glm::radians((float)angleIncrement);
	}
	return true;
}

bool EvaluateRoad::evaluateBlockades(const Configuration &configuration, const glm::vec3& position)
{
	unsigned int angleIncrement = 0;
	unsigned int length = road.roadAttributes.length;
	while (length >= configuration.minRoadLength)
	{
		do
		{
			glm::vec3 direction = glm::normalize(glm::rotate(glm::quat(glm::vec3(0, 0, road.roadAttributes.angle + glm::radians((float)angleIncrement))), glm::vec3(0.0f, 1.0f, 0.0f)));

			// FIXME: checking invariants
			if (configuration.blockadesMap == 0)
			{
				throw std::exception("configuration.blockadesMap == 0");
			}

			if (configuration.blockadesMap->castRay(position, direction, length, 0))
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
		return false;
	}

	else
	{
		road.roadAttributes.angle += glm::radians((float)angleIncrement);
	}
	return true;
}
