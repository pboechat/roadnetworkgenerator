#ifndef BRANCH_H
#define BRANCH_H

#pragma once

#include <CpuGpuCompatibility.h>
#include <RoadAttributes.h>
#include <StreetRuleAttributes.h>
#include <HighwayRuleAttributes.h>

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

//typedef Branch<StreetRuleAttributes> StreetBranch;
typedef Branch<HighwayRuleAttributes> HighwayBranch;

#endif