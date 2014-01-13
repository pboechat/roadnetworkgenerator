#ifndef ROAD_H
#define ROAD_H

#include <RoadState.h>
#include <RoadAttributes.h>

template <typename RuleAttributesType>
struct Road
{
	int delay;
	RoadAttributes roadAttributes;
	RuleAttributesType ruleAttributes;
	RoadState state;

	Road() {}
	Road(int delay, const RoadAttributes& roadAttributes, const RuleAttributesType& ruleAttributes, RoadState state) : delay(delay), roadAttributes(roadAttributes), ruleAttributes(ruleAttributes), state(state) {}

};

#endif
