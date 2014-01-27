#ifndef WORKQUEUESSET_H
#define WORKQUEUESSET_H

#include <MarshallingQueue.h>
#include <MathExtras.h>

#include <exception>

class WorkQueuesSet
{
public:
	WorkQueuesSet(MarshallingQueue* workQueues, unsigned int numWorkQueues) : workQueues(workQueues), numWorkQueues(numWorkQueues), numWorkItems(0) {}
	~WorkQueuesSet() {}

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
		if (operationCode >= numWorkQueues)
		{
			throw std::exception("operationCode >= numQueues");
		}

		workQueues[operationCode].enqueue(workItem);
		numWorkItems++;
	}

	void executeAllWorkItems(WorkQueuesSet* backQueues);

	void clear()
	{
		for (unsigned int i = 0; i < numWorkQueues; i++)
		{
			workQueues[i].clear();
		}
		numWorkItems = 0;
	}

private:
	MarshallingQueue* workQueues;
	unsigned int numWorkQueues;
	unsigned int numWorkItems;

	template<typename ProcedureType, typename WorkItemType>
	void executeAllWorkItemsInQueue(MarshallingQueue& queue, WorkQueuesSet* backQueues)
	{
		WorkItemType workItem;

		while (queue.dequeue(workItem))
		{
			ProcedureType::execute(workItem, backQueues);
			numWorkItems--;
		}
	}

};

#endif