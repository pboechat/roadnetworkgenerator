#include <ProceduresDeclarations.h>
#include <ProceduresCodes.h>

//////////////////////////////////////////////////////////////////////////
void EvaluateStreetBranch::execute(StreetBranch& branch, WorkQueuesSet* backQueues)
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
		backQueues->addWorkItem(EVALUATE_STREET, Street(0, branch.roadAttributes, branch.ruleAttributes, UNASSIGNED));
	}
}

//////////////////////////////////////////////////////////////////////////
void EvaluateHighwayBranch::execute(HighwayBranch& branch, WorkQueuesSet* backQueues)
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
		backQueues->addWorkItem(EVALUATE_HIGHWAY, Highway(0, branch.roadAttributes, branch.ruleAttributes, UNASSIGNED));
	}
}
