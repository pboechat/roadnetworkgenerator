#ifndef WORKQUEUES_H
#define WORKQUEUES_H

#include <StaticMarshallingQueue.h>
#include <MathExtras.h>

#include <exception>

class WorkQueues
{
public:
	WorkQueues(StaticMarshallingQueue** queues, unsigned int numQueues) : queues(queues), numQueues(numQueues), numWorkItems(0) {}
	~WorkQueues() {}

	inline unsigned int getNumWorkItems() const
	{
		return numWorkItems;
	}

	inline bool notEmpty() const
	{
		return numWorkItems > 0;
	}

	template<typename WorkItemType>
	void addWorkItem(unsigned int operationCode, WorkItemType& workItem)
	{
		// FIXME: checking invariants
		if (operationCode >= numQueues)
		{
			throw std::exception("operationCode >= numQueues");
		}

		queues[operationCode]->enqueue(workItem);
		numWorkItems++;
	}

	void executeAllWorkItems(WorkQueues* backQueues);

private:
	StaticMarshallingQueue** queues;
	unsigned int numQueues;
	unsigned int numWorkItems;

	template<typename ProcedureType, typename WorkItemType>
	void executeAllWorkItemsInQueue(StaticMarshallingQueue* queue, WorkQueues* backQueues)
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