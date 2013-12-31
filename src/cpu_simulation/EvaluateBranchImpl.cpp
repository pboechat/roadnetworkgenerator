#include <ProceduresDeclarations.h>
#include <ProceduresCodes.h>

//////////////////////////////////////////////////////////////////////////
void EvaluateBranch::execute(Branch& branch, WorkQueuesSet* backQueues)
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
		backQueues->addWorkItem(EVALUATE_BRANCH, branch);
	}

	// p5
	else if (branch.delay == 0)
	{
		backQueues->addWorkItem(EVALUATE_ROAD, Road(0, branch.roadAttributes, branch.ruleAttributes, UNASSIGNED));
	}
}
