#include <WorkQueues.h>
#include <Globals.h>
#include <Procedures.h>

WorkQueues::WorkQueues(unsigned int workQueuesCapacity) : mQueues(0), numWorkItems(0)
{
	unsigned int itemSize = MathExtras::max(sizeof(Road), sizeof(Branch));
	mQueues = new GenericQueue*[NUM_PROCEDURES];
	for (unsigned int i = 0; i < NUM_PROCEDURES; i++)
	{
		mQueues[i] = new GenericQueue(workQueuesCapacity, itemSize);
	}
}

WorkQueues::~WorkQueues()
{
	{
		if (mQueues != 0)
		{
			for (unsigned int i = 0; i < NUM_PROCEDURES; i++)
			{
				delete mQueues[i];
			}

			delete[] mQueues;
		}
	}
}

void WorkQueues::executeAllWorkItems(WorkQueues* backQueues)
{
	unsigned int i = 0;
	while (i < NUM_PROCEDURES)
	{
		GenericQueue* queue = mQueues[i];

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