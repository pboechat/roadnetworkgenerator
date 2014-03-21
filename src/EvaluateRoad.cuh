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
#include <GlobalVariables.cuh>

//////////////////////////////////////////////////////////////////////////
template<typename RuleAttributesType>
DEVICE_CODE void getHalfMaxObstacleDeviationAngle(Road<RuleAttributesType>& road, int& halfMaxObstacleDeviationAngle, Context* context);

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void getHalfMaxObstacleDeviationAngle(Road<HighwayRuleAttributes>& highway, int& halfMaxObstacleDeviationAngle, Context* context)
{
	halfMaxObstacleDeviationAngle = g_dConfiguration.halfMaxHighwayObstacleDeviationAngle;
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void getHalfMaxObstacleDeviationAngle(Road<StreetRuleAttributes>& street, int& halfMaxObstacleDeviationAngle, Context* context)
{
	halfMaxObstacleDeviationAngle = g_dConfiguration.halfMaxStreetObstacleDeviationAngle;
}

//////////////////////////////////////////////////////////////////////////
template<typename RuleAttributesType>
DEVICE_CODE void getMinLength(Road<RuleAttributesType>& road, unsigned int& minLength, Context* context);

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void getMinLength(Road<HighwayRuleAttributes>& highway, unsigned int& minLength, Context* context)
{
	minLength = g_dConfiguration.minHighwayLength;
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void getMinLength(Road<StreetRuleAttributes>& street, unsigned int& minLength, Context* context)
{
	minLength = g_dConfiguration.minStreetLength;
}

//////////////////////////////////////////////////////////////////////////
template<typename RuleAttributesType>
DEVICE_CODE bool evaluateWaterBodies(Road<RuleAttributesType>& road, const vml_vec2& position, Context* context);

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE bool evaluateWaterBodies(Road<HighwayRuleAttributes>& road, const vml_vec2& position, Context* context)
{
	// FIXME: checking invariants
	if (context->waterBodiesMap == 0)
	{
		THROW_EXCEPTION("context->waterBodiesMap == 0");
	}

	unsigned int minLength;
	int halfMaxObstacleDeviationAngle;

	getMinLength(road, minLength, context);
	getHalfMaxObstacleDeviationAngle(road, halfMaxObstacleDeviationAngle, context);

	bool found = false;
	unsigned int length = road.roadAttributes.length;
	int angleIncrement = 0;

	while (length >= minLength)
	{
		while (angleIncrement <= halfMaxObstacleDeviationAngle)
		{
			vml_vec2 direction = vml_normalize(vml_rotate2D(vml_vec2(0.0f, 1.0f), road.roadAttributes.angle + vml_radians((float)angleIncrement)));
			bool hit;
			vml_vec2 hitPoint;
			//CAST_RAY(g_dWaterBodiesTexture, position, direction, length, 0, hit, hitPoint);
			CAST_RAY(context->waterBodiesMap, position, direction, length, 0, hit, hitPoint);
			if (!hit)
			{
				road.state = SUCCEED;
				found = true;
				goto outside_loops;
			}
			angleIncrement++;
		}
		length--;
	}

	length = road.roadAttributes.length;
	angleIncrement = -halfMaxObstacleDeviationAngle;
	while (length >= minLength)
	{
		while (angleIncrement < 0)
		{
			vml_vec2 direction = vml_normalize(vml_rotate2D(vml_vec2(0.0f, 1.0f), road.roadAttributes.angle + vml_radians((float)angleIncrement)));
			bool hit;
			vml_vec2 hitPoint;
			//CAST_RAY(g_dWaterBodiesTexture, position, direction, length, 0, hit, hitPoint);
			CAST_RAY(context->waterBodiesMap, position, direction, length, 0, hit, hitPoint);
			if (!hit)
			{
				road.state = SUCCEED;
				found = true;
				goto outside_loops;
			}
			angleIncrement++;
		}
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
DEVICE_CODE bool evaluateWaterBodies(Road<StreetRuleAttributes>& road, const vml_vec2& position, Context* context)
{
	// FIXME: checking invariants
	if (context->waterBodiesMap == 0)
	{
		THROW_EXCEPTION("context->waterBodiesMap == 0");
	}

	unsigned int minLength;
	getMinLength(road, minLength, context);

	bool found = false;
	unsigned int length = road.roadAttributes.length;

	vml_vec2 direction = vml_normalize(vml_rotate2D(vml_vec2(0.0f, 1.0f), road.roadAttributes.angle));
	while (length >= minLength)
	{
		bool hit;
		vml_vec2 hitPoint;
		//CAST_RAY(g_dWaterBodiesTexture, position, direction, length, 0, hit, hitPoint);
		CAST_RAY(context->waterBodiesMap, position, direction, length, 0, hit, hitPoint);
		if (!hit)
		{
			road.state = SUCCEED;
			found = true;
			break;
		}
		length--;
	}

	if (!found)
	{
		road.state = FAILED;
		return false;
	}

	road.roadAttributes.length = length;

	return true;
}

//////////////////////////////////////////////////////////////////////////
template<typename RuleAttributesType>
DEVICE_CODE bool evaluateBlockades(Road<RuleAttributesType>& road, const vml_vec2& position, Context* context);

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE bool evaluateBlockades(Road<HighwayRuleAttributes>& road, const vml_vec2& position, Context* context)
{
	// FIXME: checking invariants
	if (context->blockadesMap == 0)
	{
		THROW_EXCEPTION("context->blockadesMap == 0");
	}

	unsigned int minLength;
	int halfMaxObstacleDeviationAngle;

	getMinLength(road, minLength, context);
	getHalfMaxObstacleDeviationAngle(road, halfMaxObstacleDeviationAngle, context);

	bool found = false;
	unsigned int length = road.roadAttributes.length;
	int angleIncrement = 0;

	while (length >= minLength)
	{
		while (angleIncrement <= halfMaxObstacleDeviationAngle)
		{
			vml_vec2 direction = vml_normalize(vml_rotate2D(vml_vec2(0.0f, 1.0f), road.roadAttributes.angle + vml_radians((float)angleIncrement)));

			bool hit;
			vml_vec2 hitPoint;
			//CAST_RAY(g_dBlockadesTexture, position, direction, length, 0, hit, hitPoint);
			CAST_RAY(context->blockadesMap, position, direction, length, 0, hit, hitPoint);
			if (!hit)
			{
				road.state = SUCCEED;
				found = true;
				goto outside_loops;
			}
			angleIncrement++;
		}
		length--;
	}

	length = road.roadAttributes.length;
	angleIncrement = -halfMaxObstacleDeviationAngle;
	while (length >= minLength)
	{
		while (angleIncrement < 0)
		{
			vml_vec2 direction = vml_normalize(vml_rotate2D(vml_vec2(0.0f, 1.0f), road.roadAttributes.angle + vml_radians((float)angleIncrement)));

			bool hit;
			vml_vec2 hitPoint;
			//CAST_RAY(g_dBlockadesTexture, position, direction, length, 0, hit, hitPoint);
			CAST_RAY(context->blockadesMap, position, direction, length, 0, hit, hitPoint);
			if (!hit)
			{
				road.state = SUCCEED;
				found = true;
				goto outside_loops;
			}
			angleIncrement++;
		}
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
DEVICE_CODE bool evaluateBlockades(Road<StreetRuleAttributes>& road, const vml_vec2& position, Context* context)
{
	// FIXME: checking invariants
	if (context->blockadesMap == 0)
	{
		THROW_EXCEPTION("context->blockadesMap == 0");
	}

	unsigned int minLength;
	getMinLength(road, minLength, context);

	bool found = false;
	unsigned int length = road.roadAttributes.length;

	vml_vec2 direction = vml_normalize(vml_rotate2D(vml_vec2(0.0f, 1.0f), road.roadAttributes.angle));
	while (length >= minLength)
	{
		bool hit;
		vml_vec2 hitPoint;
		//CAST_RAY(g_dBlockadesTexture, position, direction, length, 0, hit, hitPoint);
		CAST_RAY(context->blockadesMap, position, direction, length, 0, hit, hitPoint);
		if (!hit)
		{
			road.state = SUCCEED;
			found = true;
			break;
		}
		length--;
	}

	if (!found)
	{
		road.state = FAILED;
		return false;
	}

	road.roadAttributes.length = length;

	return true;
}

//////////////////////////////////////////////////////////////////////////
template<typename RuleAttributesType>
DEVICE_CODE void evaluateLocalContraints(Road<RuleAttributesType>& road, Context* context);

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void evaluateLocalContraints(Road<StreetRuleAttributes>& road, Context* context)
{
	vml_vec2 position = context->graph->vertices[road.roadAttributes.source].getPosition();

	// remove roads that cross world boundaries
	if (position.x < 0 || position.x > (float)g_dConfiguration.worldWidth ||
		position.y < 0 || position.y > (float)g_dConfiguration.worldHeight)
	{
		road.state = FAILED;
		return;
	}

	if (context->blockadesMap == 0)
	{
		road.state = SUCCEED;
		return;
	}

	evaluateBlockades(road, position, context);
}

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE void evaluateLocalContraints(Road<HighwayRuleAttributes>& road, Context* context)
{
	vml_vec2 position = context->graph->vertices[road.roadAttributes.source].getPosition();

	// remove roads that cross world boundaries
	if (position.x < 0 || position.x > (float)g_dConfiguration.worldWidth ||
		position.y < 0 || position.y > (float)g_dConfiguration.worldHeight)
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
		if (road.state == FAILED)
		{
			THROW_EXCEPTION("road.state == FAILED");
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

		if (road.state == SUCCEED)
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
		if (road.state == FAILED)
		{
			THROW_EXCEPTION("road.state == FAILED");
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

		if (road.state == SUCCEED)
		{
			backQueues[INSTANTIATE_HIGHWAY].push(road);
		}
	}
};

#endif