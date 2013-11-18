#ifndef BASEPROCEDURE_H
#define BASEPROCEDURE_H

#include <WorkItem.h>
#include <WorkQueuesManager.h>

#include <vector>

class Procedure : public WorkItem
{
public:
	virtual void execute(WorkQueuesManager<Procedure>& workQueuesManager, std::vector<Segment>& segments) = 0;

};

#endif