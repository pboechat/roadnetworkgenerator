#ifndef ROAD_H
#define ROAD_H

#include "Defines.h"
#include <RoadState.h>
#include <RoadAttributes.h>
#include <StreetRuleAttributes.h>
#include <HighwayRuleAttributes.h>

template <typename RuleAttributesType>
HOST_AND_DEVICE_CODE struct Road
{
	int delay;
	RoadAttributes roadAttributes;
	RuleAttributesType ruleAttributes;
	RoadState state;

	Road() {}
	Road(int delay, const RoadAttributes& roadAttributes, RoadState state) : delay(delay), roadAttributes(roadAttributes), state(state) {}
	Road(int delay, const RoadAttributes& roadAttributes, const RuleAttributesType& ruleAttributes, RoadState state) : delay(delay), roadAttributes(roadAttributes), ruleAttributes(ruleAttributes), state(state) {}

};

typedef Road<StreetRuleAttributes> Street;
typedef Road<HighwayRuleAttributes> Highway;

#endif
