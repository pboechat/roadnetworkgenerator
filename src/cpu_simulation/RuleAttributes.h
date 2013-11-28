#ifndef RULEATTRIBUTES_H
#define RULEATTRIBUTES_H

#include <glm/glm.hpp>

struct RuleAttributes
{
	unsigned int streetBranchDepth;
	unsigned int highwayBranchingDistance;
	unsigned int pureHighwayBranchingDistance;
	bool hasGoal;
	glm::vec3 goal;

	RuleAttributes() : streetBranchDepth(0), highwayBranchingDistance(0), pureHighwayBranchingDistance(0), hasGoal(false) {}

};

#endif