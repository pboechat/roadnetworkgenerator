#include <WorkQueuesSet.h>
#include <ProceduresDeclarations.h>
#include <ProceduresCodes.h>

void WorkQueuesSet::executeAllWorkItems(WorkQueuesSet* backQueues)
{
	unsigned int i = 0;

	while (i < numWorkQueues)
	{
		StaticMarshallingQueue& queue = workQueues[i];

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