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

void EvaluateBranch::execute(WorkQueuesManager<Procedure>& workQueuesManager, RoadNetworkGraph::Graph& roadNetworkGraph, const Configuration& configuration)
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
		workQueuesManager.addWorkItem(new EvaluateBranch(branch));
	}

	// p5
	else if (branch.delay == 0)
	{
		workQueuesManager.addWorkItem(new EvaluateRoad(Road(0, branch.roadAttributes, branch.ruleAttributes, UNASSIGNED)));
	}
}
