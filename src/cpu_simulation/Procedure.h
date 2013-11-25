#ifndef PROCEDURE_H
#define PROCEDURE_H

#include <WorkItem.h>
#include <WorkQueuesManager.h>
#include <Configuration.h>
#include <Graph.h>

struct Procedure : public WorkItem
{
	virtual void execute(WorkQueuesManager& manager, RoadNetworkGraph::Graph& graph, const Configuration& configuration) = 0;

};

#endif