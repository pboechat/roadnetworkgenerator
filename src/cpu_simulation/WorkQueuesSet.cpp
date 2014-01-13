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
			executeAllWorkItemsInQueue<EvaluateHighwayBranch, Branch<HighwayRuleAttributes> >(queue, backQueues);
			break;

		case EVALUATE_HIGHWAY:
			executeAllWorkItemsInQueue<EvaluateHighway, Road<HighwayRuleAttributes> >(queue, backQueues);
			break;

		case INSTANTIATE_HIGHWAY:
			executeAllWorkItemsInQueue<InstantiateHighway, Road<HighwayRuleAttributes> >(queue, backQueues);
			break;

		case EVALUATE_STREET_BRANCH:
			executeAllWorkItemsInQueue<EvaluateStreetBranch, Branch<StreetRuleAttributes> >(queue, backQueues);
			break;

		case EVALUATE_STREET:
			executeAllWorkItemsInQueue<EvaluateStreet, Road<StreetRuleAttributes> >(queue, backQueues);
			break;

		case INSTANTIATE_STREET:
			executeAllWorkItemsInQueue<InstantiateStreet, Road<StreetRuleAttributes> >(queue, backQueues);
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