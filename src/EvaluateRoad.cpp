#include <ProceduresDeclarations.h>
#include <ProceduresCodes.h>
#include <Globals.h>

//////////////////////////////////////////////////////////////////////////
template<typename RuleAttributesType>
void evaluateLocalContraints(Road<RuleAttributesType>& road);
//////////////////////////////////////////////////////////////////////////
template<typename RuleAttributesType>
bool evaluateWaterBodies(Road<RuleAttributesType>& road, const vml_vec2& position);
//////////////////////////////////////////////////////////////////////////
template<typename RuleAttributesType>
bool evaluateBlockades(Road<RuleAttributesType>& road, const vml_vec2& position);

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void EvaluateStreet::execute(Street& road, WorkQueuesSet* backQueues)
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
			//throw std::exception("road.state == UNASSIGNED");
			THROW_EXCEPTION("road.state == UNASSIGNED");
		}
	}

	if (road.state == FAILED)
	{
		backQueues->addWorkItem(EVALUATE_STREET, road);
	}

	else if (road.state == SUCCEED)
	{
		backQueues->addWorkItem(INSTANTIATE_STREET, road);
	}
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void EvaluateHighway::execute(Highway& road, WorkQueuesSet* backQueues)
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
			//throw std::exception("road.state == UNASSIGNED");
			THROW_EXCEPTION("road.state == UNASSIGNED");
		}
	}

	if (road.state == FAILED)
	{
		backQueues->addWorkItem(EVALUATE_HIGHWAY_BRANCH, road);
	}

	else if (road.state == SUCCEED)
	{
		backQueues->addWorkItem(INSTANTIATE_HIGHWAY, road);
	}
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void evaluateLocalContraints(Street& road)
{
	// remove streets that have exceeded max street branch depth
	if (road.ruleAttributes.branchDepth > g_dConfiguration->maxStreetBranchDepth)
	{
		road.state = FAILED;
		return;
	}

	vml_vec2 position = getPosition(g_dGraph, road.roadAttributes.source);

	// remove roads that cross world boundaries
	if (position.x < 0 || position.x > (float)g_dConfiguration->worldWidth ||
		position.y < 0 || position.y > (float)g_dConfiguration->worldHeight)
	{
		road.state = FAILED;
		return;
	}

	if (!evaluateWaterBodies(road, position))
	{
		return;
	}

	if (g_dBlockadesMap == 0)
	{
		return;
	}

	evaluateBlockades(road, position);
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void evaluateLocalContraints(Highway& road)
{
	vml_vec2 position = getPosition(g_dGraph, road.roadAttributes.source);

	// remove roads that cross world boundaries
	if (position.x < 0 || position.x > (float)g_dConfiguration->worldWidth ||
		position.y < 0 || position.y > (float)g_dConfiguration->worldHeight)
	{
		road.state = FAILED;
		return;
	}

	if (!evaluateWaterBodies(road, position))
	{
		return;
	}

	if (g_dBlockadesMap == 0)
	{
		return;
	}

	evaluateBlockades(road, position);
}

//////////////////////////////////////////////////////////////////////////
template<typename RuleAttributesType>
DEVICE_CODE bool evaluateWaterBodies(Road<RuleAttributesType>& road, const vml_vec2& position)
{
	// FIXME: checking invariants
	if (g_dWaterBodiesMap == 0)
	{
		//throw std::exception("g_waterBodiesMap == 0");
		THROW_EXCEPTION("g_waterBodiesMap == 0");
	}

	bool found = false;
	unsigned int angleIncrement = 0;
	unsigned int length = road.roadAttributes.length;

	while (length >= g_dConfiguration->minRoadLength)
	{
		do
		{
			vml_vec2 direction = vml_normalize(vml_rotate2D(vml_vec2(0.0f, 1.0f), road.roadAttributes.angle + vml_radians((float)angleIncrement)));

			if (g_dWaterBodiesMap->castRay(position, direction, length, 0))
			{
				road.state = SUCCEED;
				found = true;
				goto outside_loops;
			}

			angleIncrement++;;
		}
		while (angleIncrement <= g_dConfiguration->maxObstacleDeviationAngle);

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
template<typename RuleAttributesType>
DEVICE_CODE bool evaluateBlockades(Road<RuleAttributesType>& road, const vml_vec2& position)
{
	// FIXME: checking invariants
	if (g_dBlockadesMap == 0)
	{
		//throw std::exception("g_blockadesMap == 0");
		THROW_EXCEPTION("g_blockadesMap == 0");
	}

	bool found = false;
	unsigned int angleIncrement = 0;
	unsigned int length = road.roadAttributes.length;

	while (length >= g_dConfiguration->minRoadLength)
	{
		do
		{
			vml_vec2 direction = vml_normalize(vml_rotate2D(vml_vec2(0.0f, 1.0f), road.roadAttributes.angle + vml_radians((float)angleIncrement)));

			if (g_dBlockadesMap->castRay(position, direction, length, 0))
			{
				road.state = SUCCEED;
				found = true;
				goto outside_loops;
			}

			angleIncrement++;;
		}
		while (angleIncrement <= g_dConfiguration->maxObstacleDeviationAngle);

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