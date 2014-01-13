#include <ProceduresDeclarations.h>
#include <ProceduresCodes.h>

//////////////////////////////////////////////////////////////////////////
void EvaluateStreetBranch::execute(Branch<StreetRuleAttributes>& branch, WorkQueuesSet* backQueues)
{
	// p6
	if (branch.delay < 0)
	{
		return;
	}

	// p4
	else if (branch.delay > 0)
	{
		branch.delay--;
		backQueues->addWorkItem(EVALUATE_STREET_BRANCH, branch);
	}

	// p5
	else if (branch.delay == 0)
	{
		backQueues->addWorkItem(EVALUATE_STREET, Road<StreetRuleAttributes>(0, branch.roadAttributes, branch.ruleAttributes, UNASSIGNED));
	}
}

//////////////////////////////////////////////////////////////////////////
void EvaluateHighwayBranch::execute(Branch<HighwayRuleAttributes>& branch, WorkQueuesSet* backQueues)
{
	// p6
	if (branch.delay < 0)
	{
		return;
	}

	// p4
	else if (branch.delay > 0)
	{
		branch.delay--;
		backQueues->addWorkItem(EVALUATE_HIGHWAY_BRANCH, branch);
	}

	// p5
	else if (branch.delay == 0)
	{
		backQueues->addWorkItem(EVALUATE_HIGHWAY, Road<HighwayRuleAttributes>(0, branch.roadAttributes, branch.ruleAttributes, UNASSIGNED));
	}
}
