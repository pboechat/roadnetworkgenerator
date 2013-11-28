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

private:
	Road road;

	void evaluateLocalContraints(const Configuration& configuration, const RoadNetworkGraph::Graph& roadNetworkGraph);
	bool evaluateWaterBodies(const Configuration &configuration, const glm::vec3& position);
	bool evaluateBlockades(const Configuration &configuration, const glm::vec3& position);
	

};

#endif