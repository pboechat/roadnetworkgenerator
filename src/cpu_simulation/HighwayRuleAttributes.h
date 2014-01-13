#ifndef HIGHWAYRULEATTRIBUTES_H
#define HIGHWAYRULEATTRIBUTES_H

#include <vector_math.h>

struct HighwayRuleAttributes
{
	unsigned int branchingDistance;
	bool hasGoal;
	vml_vec2 goal;

	HighwayRuleAttributes() : branchingDistance(0), hasGoal(false) {}

};

#endif