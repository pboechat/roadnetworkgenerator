#ifndef PROCEDURE_H
#define PROCEDURE_H

#include <WorkItem.h>
#include <WorkQueuesManager.h>
#include <Configuration.h>
#include <QuadTree.h>

#include <vector>

class Procedure : public WorkItem
{
public:
	virtual void execute(WorkQueuesManager<Procedure>& workQueuesManager, QuadTree& quadtree, const Configuration& configuration) = 0;

};

#endif