#ifndef EVALUATEROAD_H
#define EVALUATEROAD_H

#include <Procedures.h>
#include <Road.h>

#include <vector_math.h>

struct EvaluateRoad : public Procedure<EvaluateRoad, Road>
{
	static void getCode()
	{
		return 1;
	}

	static void execute(Road& road, WorkQueues& queues, RoadNetworkGraph::Graph& graph, const Configuration& configuration);

private:
	static void evaluateLocalContraints(Road& road, const Configuration& configuration, const RoadNetworkGraph::Graph& roadNetworkGraph);
	static bool evaluateWaterBodies(Road& road, const Configuration& configuration, const vml_vec2& position);
	static bool evaluateBlockades(Road& road, const Configuration& configuration, const vml_vec2& position);


};

#endif