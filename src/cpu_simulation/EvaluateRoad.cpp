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

void EvaluateRoad::execute(WorkQueuesManager<Procedure>& workQueuesManager, std::vector<Segment>& segments, const Configuration& configuration)
{
	// p1, p3 and p6
	if (road.delay < 0 || road.state == FAILED)
	{
		return;
	}

	// p8
	if (road.state == UNASSIGNED)
	{
		checkLocalContraints(configuration);

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

void EvaluateRoad::checkLocalContraints(const Configuration& configuration)
{
	int angleIncrement = 0;

	do {
		glm::vec3 direction(0.0f, 1.0f, 0.0f);
		direction = glm::rotate(glm::quat(glm::vec3(0, 0, road.roadAttributes.angle + (float)angleIncrement)), direction);
		direction = glm::normalize(direction);

		if (configuration.waterBodiesMap.castRay(road.roadAttributes.start, direction, road.roadAttributes.length, 0))
		{
			road.state = SUCCEED;
			break;
		}

		angleIncrement += configuration.angleIncrement;
	} while (angleIncrement <= configuration.maxAngleIncrement);

	if (angleIncrement > configuration.maxAngleIncrement)
	{
		road.state = FAILED;
	}
}