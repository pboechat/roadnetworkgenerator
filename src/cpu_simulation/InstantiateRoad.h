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
	virtual void execute(WorkQueuesManager<Procedure>& workQueuesManager, QuadTree& quadtree, const Configuration& configuration);

private:
	Road road;

	void evaluateGlobalGoals(const Configuration& configuration, const glm::vec3& roadEnd, int* delays, RoadAttributes* roadAttributes, RuleAttributes* ruleAttributes);
	void adjustHighwayAttributes(RoadAttributes& roadAttributes, const Configuration& configuration) const;
	glm::vec3 snap(const glm::vec3& point, const Configuration &configuration, QuadTree &quadtree) const;

};

#endif