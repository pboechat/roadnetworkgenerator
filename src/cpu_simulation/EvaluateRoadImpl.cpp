#include <Procedures.h>

//////////////////////////////////////////////////////////////////////////
void evaluateLocalContraints(Road& road);
//////////////////////////////////////////////////////////////////////////
bool evaluateWaterBodies(Road& road, const vml_vec2& position);
//////////////////////////////////////////////////////////////////////////
bool evaluateBlockades(Road& road, const vml_vec2& position);

//////////////////////////////////////////////////////////////////////////
void EvaluateRoad::execute(Road& road, WorkQueues* backQueues)
{
	// p1, p3 and p6
	if (road.delay < 0 || road.state == FAILED)
	{
		return;
	}

	// p8
	if (road.state == UNASSIGNED)
	{
		evaluateLocalContraints(road);

		// FIXME: checking invariants
		if (road.state == UNASSIGNED)
		{
			throw std::exception("road.state == UNASSIGNED");
		}
	}

	if (road.state == FAILED)
	{
		backQueues->addWorkItem(EVALUATE_ROAD, road);
	}

	else if (road.state == SUCCEED)
	{
		backQueues->addWorkItem(INSTANTIATE_ROAD, road);
	}
}

//////////////////////////////////////////////////////////////////////////
void evaluateLocalContraints(Road& road)
{
	// remove streets that have exceeded max street branch depth
	if (!road.roadAttributes.highway && road.ruleAttributes.streetBranchDepth > configuration->maxStreetBranchDepth)
	{
		road.state = FAILED;
		return;
	}

	vml_vec2 position = graph->getPosition(road.roadAttributes.source);

	// remove roads that cross world boundaries
	if (position.x < 0 || position.x > (float)configuration->worldWidth ||
		position.y < 0 || position.y > (float)configuration->worldHeight)
	{
		road.state = FAILED;
		return;
	}

	if (!evaluateWaterBodies(road, position))
	{
		return;
	}

	if (configuration->blockadesMap == 0)
	{
		return;
	}

	evaluateBlockades(road, position);
}

//////////////////////////////////////////////////////////////////////////
bool evaluateWaterBodies(Road& road, const vml_vec2& position)
{
	// FIXME: checking invariants
	if (configuration->waterBodiesMap == 0)
	{
		throw std::exception("configuration->waterBodiesMap == 0");
	}

	bool found = false;
	unsigned int angleIncrement = 0;
	unsigned int length = road.roadAttributes.length;

	while (length >= configuration->minRoadLength)
	{
		do
		{
			vml_vec2 direction = vml_normalize(vml_rotate2D(vml_vec2(0.0f, 1.0f), road.roadAttributes.angle + vml_radians((float)angleIncrement)));

			if (configuration->waterBodiesMap->castRay(position, direction, length, 0))
			{
				road.state = SUCCEED;
				found = true;
				goto outside_loops;
			}

			angleIncrement++;;
		}
		while (angleIncrement <= configuration->maxObstacleDeviationAngle);

		length--;
	}

outside_loops:

	if (!found)
	{
		road.state = FAILED;
		return false;
	}

	road.roadAttributes.length = length;
	road.roadAttributes.angle += vml_radians((float)angleIncrement);
	
	return true;
}

//////////////////////////////////////////////////////////////////////////
bool evaluateBlockades(Road& road, const vml_vec2& position)
{
	// FIXME: checking invariants
	if (configuration->blockadesMap == 0)
	{
		throw std::exception("configuration->blockadesMap == 0");
	}

	bool found = false;
	unsigned int angleIncrement = 0;
	unsigned int length = road.roadAttributes.length;

	while (length >= configuration->minRoadLength)
	{
		do
		{
			vml_vec2 direction = vml_normalize(vml_rotate2D(vml_vec2(0.0f, 1.0f), road.roadAttributes.angle + vml_radians((float)angleIncrement)));

			if (configuration->blockadesMap->castRay(position, direction, length, 0))
			{
				road.state = SUCCEED;
				found = true;
				goto outside_loops;
			}

			angleIncrement++;;
		}
		while (angleIncrement <= configuration->maxObstacleDeviationAngle);

		length--;
	}

outside_loops:

	if (!found)
	{
		road.state = FAILED;
		return false;
	}

	road.roadAttributes.length = length;
	road.roadAttributes.angle += vml_radians((float)angleIncrement);

	return true;
}
