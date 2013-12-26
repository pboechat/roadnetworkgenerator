#ifndef PROCEDURE_H
#define PROCEDURE_H

#include <WorkQueues.h>
#include <Configuration.h>
#include <Graph.h>

template<typename SubType, typename ArgType, int code>
struct Procedure
{
	inline static int getCode() const
	{
		return code;
	}

	static void execute(ArgType& item, WorkQueues& queues, RoadNetworkGraph::Graph& graph, const Configuration& configuration)
	{
		SubType::execute(item, queues, graph, configuration);
	}

};

#endif