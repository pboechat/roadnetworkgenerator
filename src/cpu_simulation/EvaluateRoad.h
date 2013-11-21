#ifndef EVALUATEROAD_H
#define EVALUATEROAD_H

#include <Procedure.h>
#include <Road.h>

class EvaluateRoad : public Procedure
{
public:
	EvaluateRoad(const Road& road);

	virtual unsigned int getCode();
	virtual void execute(WorkQueuesManager<Procedure>& workQueuesManager, RoadNetwork::Graph& roadNetworkGraph, const Configuration& configuration);

private:
	Road road;

	void enforceLocalContraints(const Configuration& configuration, const RoadNetwork::Graph& roadNetworkGraph);

};

#endif