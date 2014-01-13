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
		case EVALUATE_HIGHWAY_BRANCH:
			executeAllWorkItemsInQueue<EvaluateHighwayBranch, HighwayBranch>(queue, backQueues);
			break;

		case EVALUATE_HIGHWAY:
			executeAllWorkItemsInQueue<EvaluateHighway, Highway>(queue, backQueues);
			break;

		case INSTANTIATE_HIGHWAY:
			executeAllWorkItemsInQueue<InstantiateHighway, Highway>(queue, backQueues);
			break;

		case EVALUATE_STREET_BRANCH:
			executeAllWorkItemsInQueue<EvaluateStreetBranch, StreetBranch>(queue, backQueues);
			break;

		case EVALUATE_STREET:
			executeAllWorkItemsInQueue<EvaluateStreet, Street>(queue, backQueues);
			break;

		case INSTANTIATE_STREET:
			executeAllWorkItemsInQueue<InstantiateStreet, Street>(queue, backQueues);
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