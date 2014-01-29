#ifndef BRANCH_CUH
#define BRANCH_CUH

#include "Defines.h"
#include <RoadAttributes.cuh>
#include <StreetRuleAttributes.cuh>
#include <HighwayRuleAttributes.cuh>

template<typename RuleAttributesType>
struct Branch
{
	int delay;
	RoadAttributes roadAttributes;
	RuleAttributesType ruleAttributes;

	HOST_AND_DEVICE_CODE Branch() {}
	HOST_AND_DEVICE_CODE Branch(int delay, const RoadAttributes& roadAttributes, const RuleAttributesType& ruleAttributes) : delay(delay), roadAttributes(roadAttributes), ruleAttributes(ruleAttributes) {}
	
	HOST_AND_DEVICE_CODE Branch& operator = (const Branch& other)
	{
		delay = other.delay;
		roadAttributes = other.roadAttributes;
		ruleAttributes = other.ruleAttributes;
		return *this;
	}

};

typedef Branch<StreetRuleAttributes> StreetBranch;
typedef Branch<HighwayRuleAttributes> HighwayBranch;

#endif