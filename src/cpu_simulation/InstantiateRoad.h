#ifndef INSTANTIATEROAD_H
#define INSTANTIATEROAD_H

#include <Procedure.h>
#include <Road.h>

#include <glm/glm.hpp>

class InstantiateRoad : public Procedure
{
public:
	InstantiateRoad(const Road& road);

	virtual unsigned int getCode();
	virtual void execute(WorkQueuesManager<Procedure>& workQueuesManager, std::vector<Segment>& segments, const Configuration& configuration);

private:
	Road road;

	void evaluateGlobalGoals(const Configuration& configuration, const glm::vec3& roadEnd, int* delays, RoadAttributes* roadAttributes, RuleAttributes* ruleAttributes);

};

#endif