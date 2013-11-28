#ifndef WORKQUEUESMANAGER_H
#define WORKQUEUESMANAGER_H

#include <static_alloc_queue.h>
#include <WorkItem.h>
#include <Configuration.h>
#include <Graph.h>

class WorkQueuesManager
{
public:
	WorkQueuesManager(unsigned int numberOfWorkQueues, unsigned int maxWorkQueueCapacity);
	~WorkQueuesManager();

	inline unsigned int size() const
	{
		return workItemsCounter;
	}

	inline bool notEmpty() const
	{
		return workItemsCounter > 0;
	}

	template<typename T>
	void addWorkItem(const T& workItem)
	{
		workQueues[workItem.getCode()]->enqueue(workItem);
		workItemsCounter++;
	}

	void executeAllWorkItems(WorkQueuesManager& manager, RoadNetworkGraph::Graph& graph, const Configuration& configuration);

private:
	static_alloc_queue** workQueues;
	unsigned int workQueuesCounter;
	unsigned int workItemsCounter;

	template<typename T>
	bool executeWorkItem(static_alloc_queue* workQueue, WorkQueuesManager& manager, RoadNetworkGraph::Graph& graph, const Configuration& configuration)
	{
		T workItem;
		if (workQueue->dequeue(workItem))
		{
			workItem.execute(manager, graph, configuration);
			return true;
		}
		return false;
	}
	
};

#endif