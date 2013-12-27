#include <WorkQueues.h>
#include <Globals.h>
#include <Procedures.h>

void WorkQueues::executeAllWorkItems(WorkQueues* backQueues)
{
	unsigned int i = 0;
	while (i < numQueues)
	{
		StaticMarshallingQueue* queue = queues[i];

		switch (i)
		{
		case EVALUATE_BRANCH:
			executeAllWorkItemsInQueue<EvaluateBranch, Branch>(queue, backQueues);
			break;

		case EVALUATE_ROAD:
			executeAllWorkItemsInQueue<EvaluateRoad, Road>(queue, backQueues);
			break;

		case INSTANTIATE_ROAD:
			executeAllWorkItemsInQueue<InstantiateRoad, Road>(queue, backQueues);
			break;

		default:
			// FIXME: checking invariants
			throw std::exception("unknown procedure");
		}

		i++;
	}

	// FIXME: checking invariants
	if (numWorkItems > 0)
	{
		throw std::exception("numWorkItems > 0");
	}
}