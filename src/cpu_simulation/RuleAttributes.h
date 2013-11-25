#ifndef RULEATTRIBUTES_H
#define RULEATTRIBUTES_H

struct RuleAttributes
{
	unsigned int streetBranchDepth;
	unsigned int highwayBranchingDistance;
	unsigned int pureHighwayBranchingDistance;
	bool hasGoal;
	float goalDistance;

	RuleAttributes() : streetBranchDepth(0), highwayBranchingDistance(0), pureHighwayBranchingDistance(0), hasGoal(false), goalDistance(0) {}

	RuleAttributes& operator = (const RuleAttributes& other)
	{
		streetBranchDepth = other.streetBranchDepth;
		highwayBranchingDistance = other.highwayBranchingDistance;
		pureHighwayBranchingDistance = other.pureHighwayBranchingDistance;
		hasGoal = other.hasGoal;
		goalDistance = other.goalDistance;
		return *this;
	}

};

#endif