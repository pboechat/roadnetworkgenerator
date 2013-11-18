#ifndef BRANCH_H
#define BRANCH_H

#include <RoadAttributes.h>
#include <RuleAttributes.h>

#include <glm/glm.hpp>

struct Branch
{
	int delay;
	RoadAttributes roadAttributes;
	RuleAttributes ruleAttributes;

	Branch(int delay, const RoadAttributes& roadAttributes, const RuleAttributes& ruleAttributes) : delay(delay), roadAttributes(roadAttributes), ruleAttributes(ruleAttributes) {}
	~Branch() {}

};

#endif