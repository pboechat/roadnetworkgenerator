#ifndef PROCEDURE_H
#define PROCEDURE_H

#include <WorkItem.h>
#include <WorkQueuesManager.h>
#include <ImageMap.h>

#include <vector>

class Procedure : public WorkItem
{
public:
	virtual void execute(WorkQueuesManager<Procedure>& workQueuesManager, std::vector<Segment>& segments, ImageMap& populationDensityMap, ImageMap& waterBodiesMap) = 0;

};

#endif