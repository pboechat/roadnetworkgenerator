#ifndef EVALUATEROAD_H
#define EVALUATEROAD_H

#include <Procedure.h>
#include <Road.h>

#define EVALUATE_ROAD_CODE 2

struct EvaluateRoad : public Procedure
{
	EvaluateRoad();
	EvaluateRoad(const Road& road);

	virtual unsigned int getCode() const;
	virtual void execute(WorkQueuesManager& manager, RoadNetworkGraph::Graph& graph, const Configuration& configuration);
	EvaluateRoad& operator = (const EvaluateRoad& other);

private:
	Road road;

	void evaluateLocalContraints(const Configuration& configuration, const RoadNetworkGraph::Graph& roadNetworkGraph);

};

#endif