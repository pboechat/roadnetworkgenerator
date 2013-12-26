#ifndef BRANCH_H
#define BRANCH_H

#include <RoadAttributes.h>
#include <RuleAttributes.h>

struct Branch
{
	int delay;
	RoadAttributes roadAttributes;
	RuleAttributes ruleAttributes;

	Branch() {}
	Branch(int delay, const RoadAttributes& roadAttributes, const RuleAttributes& ruleAttributes) : delay(delay), roadAttributes(roadAttributes), ruleAttributes(ruleAttributes) {}

};

#endif