#ifndef HIGHWAYRULEATTRIBUTES_H
#define HIGHWAYRULEATTRIBUTES_H

#include "Defines.h"
#include <vector_math.h>

HOST_AND_DEVICE_CODE struct HighwayRuleAttributes
{
	unsigned int branchingDistance;
	bool hasGoal;
	vml_vec2 goal;

	HighwayRuleAttributes() : branchingDistance(0), hasGoal(false) {}

};

#endif