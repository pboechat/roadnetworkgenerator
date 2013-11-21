#ifndef PROCEDURE_H
#define PROCEDURE_H

#include <WorkItem.h>
#include <WorkQueuesManager.h>
#include <Configuration.h>
#include <RoadNetwork.h>

#include <vector>

class Procedure : public WorkItem
{
public:
	virtual void execute(WorkQueuesManager<Procedure>& workQueuesManager, RoadNetwork::Graph& roadNetworkGraph, const Configuration& configuration) = 0;

};

#endif