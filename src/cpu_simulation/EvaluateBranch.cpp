#include <EvaluateBranch.h>
#include <EvaluateRoad.h>
#include <Road.h>

EvaluateBranch::EvaluateBranch(const Branch& branch) : branch(branch)
{
}

unsigned int EvaluateBranch::getCode()
{
	return 1;
}

void EvaluateBranch::execute(WorkQueuesManager<Procedure>& workQueuesManager, QuadTree& quadtree, const Configuration& configuration)
{
	// p4
	if (branch.delay > 0)
	{
		branch.delay--;
		workQueuesManager.addWorkItem(new EvaluateBranch(branch));
	}

	// p5
	else if (branch.delay == 0)
	{
		workQueuesManager.addWorkItem(new EvaluateRoad(Road(0, branch.roadAttributes, branch.ruleAttributes, UNASSIGNED)));
	}

	// p6
	else
	{
		// don't do nothing
	}
}
