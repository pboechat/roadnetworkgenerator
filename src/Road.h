#ifndef ROAD_H
#define ROAD_H

#pragma once

#include <CpuGpuCompatibility.h>
#include <RoadState.h>
#include <RoadAttributes.h>
#include <StreetRuleAttributes.h>
#include <HighwayRuleAttributes.h>

template <typename RuleAttributesType>
struct Road
{
	int delay;
	RoadAttributes roadAttributes;
	RuleAttributesType ruleAttributes;
	RoadState state;

	HOST_AND_DEVICE_CODE Road() {}
	HOST_AND_DEVICE_CODE Road(int delay, const RoadAttributes& roadAttributes, RoadState state) : delay(delay), roadAttributes(roadAttributes), state(state) {}
	HOST_AND_DEVICE_CODE Road(int delay, const RoadAttributes& roadAttributes, const RuleAttributesType& ruleAttributes, RoadState state) : delay(delay), roadAttributes(roadAttributes), ruleAttributes(ruleAttributes), state(state) {}
	HOST_AND_DEVICE_CODE ~Road() {}

	HOST_AND_DEVICE_CODE Road& operator = (const Road& other)
	{
		delay = other.delay;
		roadAttributes = other.roadAttributes;
		ruleAttributes = other.ruleAttributes;
		state = other.state;
		return *this;
	}

};

typedef Road<StreetRuleAttributes> Street;
typedef Road<HighwayRuleAttributes> Highway;

#endif
