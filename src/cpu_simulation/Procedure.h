#ifndef PROCEDURE_H
#define PROCEDURE_H

#include <WorkItem.h>
#include <WorkQueuesManager.h>
#include <Configuration.h>
#include <Graph.h>

#include <vector>

class Procedure : public WorkItem
{
public:
	virtual void execute(WorkQueuesManager<Procedure>& workQueuesManager, RoadNetworkGraph::Graph& roadNetworkGraph, const Configuration& configuration) = 0;

};

#endif