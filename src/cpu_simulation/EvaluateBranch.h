#ifndef EVALUATEBRANCH_H
#define EVALUATEBRANCH_H

#include <Procedure.h>
#include <Branch.h>

struct EvaluateBranch : public Procedure<EvaluateBranch, Branch>
{
	static void execute(Branch& branch, WorkQueues& queues, RoadNetworkGraph::Graph& graph, const Configuration& configuration);

};

#endif