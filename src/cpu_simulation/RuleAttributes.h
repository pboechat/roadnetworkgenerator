#ifndef RULEATTRIBUTES_H
#define RULEATTRIBUTES_H

struct RuleAttributes
{
	unsigned int streetBranchDepth;
	unsigned int highwayBranchingDistance;
	unsigned int pureHighwayBranchingDistance;
	int highwayGoalDistance;

	RuleAttributes() : streetBranchDepth(0), highwayBranchingDistance(0), pureHighwayBranchingDistance(0), highwayGoalDistance(0) {}
	~RuleAttributes() {}

	RuleAttributes& operator =(const RuleAttributes& other)
	{
		streetBranchDepth = other.streetBranchDepth;
		highwayBranchingDistance = other.highwayBranchingDistance;
		pureHighwayBranchingDistance = other.pureHighwayBranchingDistance;
		highwayGoalDistance = other.highwayGoalDistance;
		return *this;
	}

};

#endif