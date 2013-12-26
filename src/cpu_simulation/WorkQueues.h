#ifndef WORKQUEUES_H
#define WORKQUEUES_H

#include <GenericQueue.h>
#include <MathExtras.h>

class WorkQueues
{
public:
	WorkQueues(unsigned int workQueuesCapacity);
	~WorkQueues();

	inline unsigned int getNumWorkItems() const
	{
		return numWorkItems;
	}

	inline bool notEmpty() const
	{
		return numWorkItems > 0;
	}

	template<typename WorkItemType>
	void addWorkItem(int operationCode, WorkItemType& workItem)
	{
		mQueues[operationCode]->enqueue(workItem);
		numWorkItems++;
	}

	void executeAllWorkItems(WorkQueues* backQueues);

private:
	GenericQueue** mQueues;
	unsigned int numWorkItems;

	template<typename ProcedureType, typename WorkItemType>
	void executeAllWorkItemsInQueue(GenericQueue* queue, WorkQueues* backQueues)
	{
		WorkItemType workItem;
		while (queue->dequeue(workItem))
		{
			ProcedureType::execute(workItem, backQueues);
			numWorkItems--;
		}
	}

};

#endif