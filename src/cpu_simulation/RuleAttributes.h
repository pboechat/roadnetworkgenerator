#ifndef RULEATTRIBUTES_H
#define RULEATTRIBUTES_H

struct RuleAttributes
{
	int streetBranchDepth;
	int highwayBranchingDistance;
	int pureHighwayBranchingDistance;

	RuleAttributes() : streetBranchDepth(0), highwayBranchingDistance(0), pureHighwayBranchingDistance(0) {}
	~RuleAttributes() {}

	RuleAttributes& operator =(const RuleAttributes& other)
	{
		streetBranchDepth = other.streetBranchDepth;
		highwayBranchingDistance = other.highwayBranchingDistance;
		pureHighwayBranchingDistance = other.pureHighwayBranchingDistance;
		return *this;
	}

};

#endif