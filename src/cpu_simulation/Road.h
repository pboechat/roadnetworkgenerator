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

	Road(int delay, const RoadAttributes& roadAttributes, const RuleAttributes& ruleAttributes, RoadState state) : delay(delay), roadAttributes(roadAttributes), ruleAttributes(ruleAttributes), state(state) {}
	~Road() {}

};

#endif
