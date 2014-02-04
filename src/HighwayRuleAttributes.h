#ifndef HIGHWAYRULEATTRIBUTES_CUH
#define HIGHWAYRULEATTRIBUTES_CUH

#pragma once

#include <CpuGpuCompatibility.h>
#include <VectorMath.h>

struct HighwayRuleAttributes
{
	unsigned int branchingDistance;
	bool hasGoal;
	unsigned int branchDepth;
	vec2FieldDeclaration(Goal, HOST_AND_DEVICE_CODE)

	HOST_AND_DEVICE_CODE HighwayRuleAttributes() : branchingDistance(0), hasGoal(false), branchDepth(0) {}
	HOST_AND_DEVICE_CODE ~HighwayRuleAttributes() {}

	HOST_AND_DEVICE_CODE HighwayRuleAttributes& operator = (const HighwayRuleAttributes& other)
	{
		branchingDistance = other.branchingDistance;
		hasGoal = other.hasGoal;
		branchDepth = other.branchDepth;
		setGoal(other.getGoal());
		return *this;
	}

};

#endif