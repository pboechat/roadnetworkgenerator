#ifndef HIGHWAYRULEATTRIBUTES_CUH
#define HIGHWAYRULEATTRIBUTES_CUH

#include "Defines.h"

#include <vector_math.h>

struct HighwayRuleAttributes
{
	unsigned int branchingDistance;
	bool hasGoal;
	vec2FieldDeclaration(Goal, HOST_AND_DEVICE_CODE)

	HOST_AND_DEVICE_CODE HighwayRuleAttributes() : branchingDistance(0), hasGoal(false) {}
	HOST_AND_DEVICE_CODE ~HighwayRuleAttributes() {}

	HOST_AND_DEVICE_CODE HighwayRuleAttributes& operator = (const HighwayRuleAttributes& other)
	{
		branchingDistance = other.branchingDistance;
		hasGoal = other.hasGoal;
		setGoal(other.getGoal());
		return *this;
	}

};

#endif