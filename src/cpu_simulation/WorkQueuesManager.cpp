#include <WorkQueuesManager.h>
#include <Procedure.h>
#include <EvaluateBranch.h>
#include <EvaluateRoad.h>
#include <InstantiateRoad.h>

#ifndef max
#define max(a, b) (((a) > (b)) ? (a) : (b))
#endif

WorkQueuesManager::WorkQueuesManager(unsigned int numberOfWorkQueues, unsigned int workQueueCapacity) : workQueues(0), workQueuesCounter(numberOfWorkQueues), workItemsCounter(0)
{
	unsigned int itemSize = max(sizeof(EvaluateBranch), max(sizeof(EvaluateRoad), sizeof(InstantiateRoad)));

	workQueues = new static_alloc_queue*[workQueuesCounter];
	for (unsigned int i = 0; i < workQueuesCounter; i++)
	{
		workQueues[i] = new static_alloc_queue(workQueueCapacity, itemSize);
	}
}

WorkQueuesManager::~WorkQueuesManager()
{
	if (workQueues != 0)
	{
		for (unsigned int i = 0; i < workQueuesCounter; i++)
		{
			delete workQueues[i];
		}
		delete[] workQueues;
	}
}

void WorkQueuesManager::executeAllWorkItems(WorkQueuesManager& manager, RoadNetworkGraph::Graph& graph, const Configuration& configuration)
{
	unsigned int currentWorkQueueIndex = 0;

	do
	{
		static_alloc_queue* currentWorkQueue = workQueues[currentWorkQueueIndex];

		do
		{
			if (currentWorkQueueIndex == INSTANTIATE_ROAD_CODE) 
			{
				if (executeWorkItem<InstantiateRoad>(currentWorkQueue, manager, graph, configuration))
				{
					workItemsCounter--;
				}
				
				else
				{
					break;
				}
			}

			else if (currentWorkQueueIndex == EVALUATE_BRANCH_CODE)
			{
				if (executeWorkItem<EvaluateBranch>(currentWorkQueue, manager, graph, configuration))
				{
					workItemsCounter--;
				}

				else
				{
					break;
				}
			}

			else if (currentWorkQueueIndex == EVALUATE_ROAD_CODE)
			{
				if (executeWorkItem<EvaluateRoad>(currentWorkQueue, manager, graph, configuration))
				{
					workItemsCounter--;
				}

				else
				{
					break;
				}
			}

			else
			{
				// FIXME: checking invariants
				throw std::exception("invalid current work queue");
			}
		}
		while (true);
	}
	while (++currentWorkQueueIndex < workQueuesCounter);

	// FIXME: checking invariants
	if (workItemsCounter > 0)
	{
		throw std::exception("workItemsCounter > 0");
	}
}