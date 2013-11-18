#ifndef WORKQUEUESMANAGER_H
#define WORKQUEUESMANAGER_H

#include <Segment.h>

#include <map>
#include <deque>

class Procedure;

template <class WorkItem>
class WorkQueuesManager
{
public:
	WorkQueuesManager();

	void addWorkItem(WorkItem* workItem);
	bool notEmpty() const;
	void resetCursors();
	bool nextWorkQueue();
	WorkItem* popWorkItem();

private:
	typedef std::deque<WorkItem*> WorkQueue;

	std::map<unsigned int, WorkQueue> workQueues;
	unsigned int workItemsCounter;
	typename std::map<unsigned int, WorkQueue>::iterator workQueueCursor;

};

template<class WorkItem>
WorkQueuesManager<WorkItem>::WorkQueuesManager() : workItemsCounter(0)
{
}

template<class WorkItem>
void WorkQueuesManager<WorkItem>::addWorkItem(WorkItem* workItem)
{
	unsigned int code = workItem->getCode();
	std::map<unsigned int, std::deque<WorkItem*> >::iterator i;

	if ((i = workQueues.find(code)) == workQueues.end())
	{
		std::deque<WorkItem*> workQueue;
		workQueue.push_back(workItem);
		workQueues.insert(std::make_pair(code, workQueue));
	}

	else
	{
		std::deque<WorkItem*>& workQueue = i->second;
		workQueue.push_back(workItem);
	}

	workItemsCounter++;
}

template<class WorkItem>
bool WorkQueuesManager<WorkItem>::notEmpty() const
{
	return workItemsCounter > 0;
}

template<class WorkItem>
void WorkQueuesManager<WorkItem>::resetCursors()
{
	workQueueCursor = workQueues.begin();
}

template<class WorkItem>
bool WorkQueuesManager<WorkItem>::nextWorkQueue()
{
	if (workQueueCursor == workQueues.end())
	{
		return false;
	}

	workQueueCursor++;

	if (workQueueCursor == workQueues.end())
	{
		return false;
	}

	return true;
}

template<class WorkItem>
WorkItem* WorkQueuesManager<WorkItem>::popWorkItem()
{
	if (workQueueCursor == workQueues.end())
	{
		return 0;
	}

	if (workQueueCursor->second.empty())
	{
		return 0;
	}

	WorkItem* workItem = workQueueCursor->second.front();
	workQueueCursor->second.pop_front();
	workItemsCounter--;
	return workItem;
}

#endif