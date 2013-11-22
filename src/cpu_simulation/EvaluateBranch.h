#ifndef EVALUATEBRANCH_H
#define EVALUATEBRANCH_H

#include <Procedure.h>
#include <Branch.h>

class EvaluateBranch : public Procedure
{
public:
	EvaluateBranch(const Branch& branch);

	virtual unsigned int getCode();
	virtual void execute(WorkQueuesManager<Procedure>& workQueuesManager, RoadNetworkGraph::Graph& roadNetworkGraph, const Configuration& configuration);

private:
	Branch branch;

};

#endif