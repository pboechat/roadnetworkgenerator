#include <EvaluateRoad.h>
#include <InstantiateRoad.h>

#include <exception>

EvaluateRoad::EvaluateRoad(const Road& road) : road(road)
{
}

unsigned int EvaluateRoad::getCode()
{
	return 2;
}

void EvaluateRoad::execute(WorkQueuesManager<Procedure>& workQueuesManager, std::vector<Segment>& segments, ImageMap& populationDensityMap, ImageMap& waterBodiesMap)
{
	// p1, p3 and p6
	if (road.delay < 0 || road.state == FAILED)
	{
		return;
	}

	// p8
	if (road.state == UNASSIGNED)
	{
		checkLocalContraints(waterBodiesMap);

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

void EvaluateRoad::checkLocalContraints(ImageMap& waterBodiesMap)
{
	//waterBodiesMap.sample(road.start);
	road.state = SUCCEED;
}