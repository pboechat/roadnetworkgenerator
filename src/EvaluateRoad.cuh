#ifndef EVALUATEROAD_CUH
#define EVALUATEROAD_CUH

#pragma once

#include <CpuGpuCompatibility.h>
#include <Road.h>
#include <Branch.h>
#include <ProceduresCodes.h>
#include <Context.cuh>
#include <WorkQueue.cuh>
#include <GraphFunctions.cuh>
#include <ImageMapFunctions.cuh>

//////////////////////////////////////////////////////////////////////////
template<typename RuleAttributesType>
DEVICE_CODE bool evaluateWaterBodies(Road<RuleAttributesType>& road, const vml_vec2& position, Context* context)
{
	// FIXME: checking invariants
	if (context->waterBodiesMap == 0)
	{
		THROW_EXCEPTION("context->waterBodiesMap == 0");
	}

	bool found = false;
	unsigned int angleIncrement = 0;
	unsigned int length = road.roadAttributes.length;

	while (length >= context->configuration->minRoadLength)
	{
		do
		{
			vml_vec2 direction = vml_normalize(vml_rotate2D(vml_vec2(0.0f, 1.0f), road.roadAttributes.angle + vml_radians((float)angleIncrement)));

			bool hit;
			vml_vec2 hitPoint;
			CAST_RAY(waterBodiesTexture, position, direction, length, 0, hit, hitPoint);
			if (!hit)
			{
				road.state = SUCCEED;
				found = true;
				goto outside_loops;
			}

			angleIncrement++;;
		}
		while (angleIncrement <= context->configuration->maxObstacleDeviationAngle);

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
DEVICE_CODE bool evaluateBlockades(Road<RuleAttributesType>& road, const vml_vec2& position, Context* context)
{
	// FIXME: checking invariants
	if (context->blockadesMap == 0)
	{
		THROW_EXCEPTION("context->blockadesMap == 0");
	}

	bool found = false;
	unsigned int angleIncrement = 0;
	unsigned int length = road.roadAttributes.length;

	while (length >= context->configuration->minRoadLength)
	{
		do
		{
			vml_vec2 direction = vml_normalize(vml_rotate2D(vml_vec2(0.0f, 1.0f), road.roadAttributes.angle + vml_radians((float)angleIncrement)));

			bool hit;
			vml_vec2 hitPoint;
			CAST_RAY(blockadesTexture, position, direction, length, 0, hit, hitPoint);
			if (!hit)
			{
				road.state = SUCCEED;
				found = true;
				goto outside_loops;
			}

			angleIncrement++;;
		}
		while (angleIncrement <= context->configuration->maxObstacleDeviationAngle);

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
DEVICE_CODE void evaluateLocalContraints(Road<RuleAttributesType>& road, Context* context);

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void evaluateLocalContraints(Road<StreetRuleAttributes>& road, Context* context)
{
	// remove streets that have exceeded max street branch depth
	if (road.ruleAttributes.branchDepth > context->configuration->maxStreetBranchDepth)
	{
		road.state = FAILED;
		return;
	}

	vml_vec2 position = context->graph->vertices[road.roadAttributes.source].getPosition();

	// remove roads that cross world boundaries
	if (position.x < 0 || position.x > (float)context->configuration->worldWidth ||
		position.y < 0 || position.y > (float)context->configuration->worldHeight)
	{
		road.state = FAILED;
		return;
	}

	if (!evaluateWaterBodies(road, position, context))
	{
		return;
	}

	if (context->blockadesMap == 0)
	{
		return;
	}

	evaluateBlockades(road, position, context);
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void evaluateLocalContraints(Road<HighwayRuleAttributes>& road, Context* context)
{
	// remove highways that have exceeded max highway branch depth
	if (road.ruleAttributes.branchDepth > context->configuration->maxHighwayBranchDepth)
	{
		road.state = FAILED;
		return;
	}

	vml_vec2 position = context->graph->vertices[road.roadAttributes.source].getPosition();

	// remove roads that cross world boundaries
	if (position.x < 0 || position.x > (float)context->configuration->worldWidth ||
		position.y < 0 || position.y > (float)context->configuration->worldHeight)
	{
		road.state = FAILED;
		return;
	}

	if (!evaluateWaterBodies(road, position, context))
	{
		return;
	}

	if (context->blockadesMap == 0)
	{
		return;
	}

	evaluateBlockades(road, position, context);
}

//////////////////////////////////////////////////////////////////////////
struct EvaluateStreet
{
	//////////////////////////////////////////////////////////////////////////
	static DEVICE_CODE void execute(Street& road, Context* context, WorkQueue* backQueues)
	{
		// p1, p3 and p6
		if (road.delay < 0 || road.state == FAILED)
		{
			return;
		}

		// p8
		if (road.state == UNASSIGNED)
		{
			evaluateLocalContraints(road, context);

			// FIXME: checking invariants
			if (road.state == UNASSIGNED)
			{
				THROW_EXCEPTION("road.state == UNASSIGNED");
			}
		}

		if (road.state == FAILED)
		{
			backQueues[EVALUATE_STREET].push(road);
		}

		else if (road.state == SUCCEED)
		{
			backQueues[INSTANTIATE_STREET].push(road);
		}
	}
};

//////////////////////////////////////////////////////////////////////////
struct EvaluateHighway
{
	//////////////////////////////////////////////////////////////////////////
	static DEVICE_CODE void execute(Highway& road, Context* context, WorkQueue* backQueues)
	{
		// p1, p3 and p6
		if (road.delay < 0 || road.state == FAILED)
		{
			return;
		}

		// p8
		if (road.state == UNASSIGNED)
		{
			evaluateLocalContraints(road, context);

			// FIXME: checking invariants
			if (road.state == UNASSIGNED)
			{
				THROW_EXCEPTION("road.state == UNASSIGNED");
			}
		}

		if (road.state == FAILED)
		{
			backQueues[EVALUATE_HIGHWAY_BRANCH].push(road);
		}

		else if (road.state == SUCCEED)
		{
			backQueues[INSTANTIATE_HIGHWAY].push(road);
		}
	}
};

#endif