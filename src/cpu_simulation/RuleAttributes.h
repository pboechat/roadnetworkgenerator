#ifndef RULEATTRIBUTES_H
#define RULEATTRIBUTES_H

#include <vector_math.h>

struct RuleAttributes
{
	unsigned int streetBranchDepth;
	unsigned int highwayBranchingDistance;
	unsigned int pureHighwayBranchingDistance;
	bool hasGoal;
	vml_vec2 goal;

	RuleAttributes() : streetBranchDepth(0), highwayBranchingDistance(0), pureHighwayBranchingDistance(0), hasGoal(false) {}

};

#endif