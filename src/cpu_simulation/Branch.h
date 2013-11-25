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

	Branch() {}
	Branch(int delay, const RoadAttributes& roadAttributes, const RuleAttributes& ruleAttributes) : delay(delay), roadAttributes(roadAttributes), ruleAttributes(ruleAttributes) {}
	
	Branch& operator = (const Branch& other)
	{
		delay = other.delay;
		roadAttributes = other.roadAttributes;
		ruleAttributes = other.ruleAttributes;
		return *this;
	}

};

#endif