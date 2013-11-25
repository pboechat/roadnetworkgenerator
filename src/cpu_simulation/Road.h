#ifndef ROAD_H
#define ROAD_H

#include <RoadAttributes.h>
#include <RuleAttributes.h>

#include <glm/glm.hpp>

enum RoadState
{
	UNASSIGNED,
	SUCCEED,
	FAILED
};

struct Road
{
	int delay;
	RoadAttributes roadAttributes;
	RuleAttributes ruleAttributes;
	RoadState state;

	Road() {}
	Road(int delay, const RoadAttributes& roadAttributes, const RuleAttributes& ruleAttributes, RoadState state) : delay(delay), roadAttributes(roadAttributes), ruleAttributes(ruleAttributes), state(state) {}

	Road& operator = (const Road& other)
	{
		delay = other.delay;
		roadAttributes = other.roadAttributes;
		ruleAttributes = other.ruleAttributes;
		state = other.state;
		return *this;
	}
};

#endif
