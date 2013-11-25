#ifndef EVALUATEBRANCH_H
#define EVALUATEBRANCH_H

#include <Procedure.h>
#include <Branch.h>

#define EVALUATE_BRANCH_CODE 1

struct EvaluateBranch : public Procedure
{
	EvaluateBranch();
	EvaluateBranch(const Branch& branch);

	virtual unsigned int getCode() const;
	virtual void execute(WorkQueuesManager& manager, RoadNetworkGraph::Graph& graph, const Configuration& configuration);
	EvaluateBranch& operator = (const EvaluateBranch& other);

private:
	Branch branch;

};

#endif