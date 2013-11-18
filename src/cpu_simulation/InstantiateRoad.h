#ifndef INSTANTIATEROAD_H
#define INSTANTIATEROAD_H

#include <Procedure.h>
#include <Road.h>

class InstantiateRoad : public Procedure
{
public:
	InstantiateRoad(const Road& road);

	virtual unsigned int getCode();
	virtual void execute(WorkQueuesManager<Procedure>& workQueuesManager, std::vector<Segment>& segments, ImageMap& populationDensityMap, ImageMap& waterBodiesMap);
	void evaluateGlobalGoals(int* delays, RoadAttributes* roadAttributes, RuleAttributes* ruleAttributes);

private:
	Road road;

};

#endif