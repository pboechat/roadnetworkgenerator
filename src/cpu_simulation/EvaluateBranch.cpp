#include <EvaluateBranch.h>
#include <EvaluateRoad.h>
#include <Road.h>

EvaluateBranch::EvaluateBranch()
{
}

EvaluateBranch::EvaluateBranch(const Branch& branch) : branch(branch)
{
}

unsigned int EvaluateBranch::getCode() const
{
	return EVALUATE_BRANCH_CODE;
}

void EvaluateBranch::execute(WorkQueuesManager& manager, RoadNetworkGraph::Graph& graph, const Configuration& configuration)
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
		manager.addWorkItem(EvaluateBranch(branch));
	}

	// p5
	else if (branch.delay == 0)
	{
		manager.addWorkItem(EvaluateRoad(Road(0, branch.roadAttributes, branch.ruleAttributes, UNASSIGNED)));
	}
}

EvaluateBranch& EvaluateBranch::operator = (const EvaluateBranch& other)
{
	branch = other.branch;
	return *this;
}