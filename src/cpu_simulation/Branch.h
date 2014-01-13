#ifndef BRANCH_H
#define BRANCH_H

#include <RoadAttributes.h>

template<typename RuleAttributesType>
struct Branch
{
	int delay;
	RoadAttributes roadAttributes;
	RuleAttributesType ruleAttributes;

	Branch() {}
	Branch(int delay, const RoadAttributes& roadAttributes, const RuleAttributesType& ruleAttributes) : delay(delay), roadAttributes(roadAttributes), ruleAttributes(ruleAttributes) {}

};

#endif