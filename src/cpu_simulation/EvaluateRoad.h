#ifndef EVALUATEROAD_H
#define EVALUATEROAD_H

#include <Procedure.h>
#include <Road.h>

class EvaluateRoad : public Procedure
{
public:
	EvaluateRoad(const Road& road);

	virtual unsigned int getCode();
	virtual void execute(WorkQueuesManager<Procedure>& workQueuesManager, std::vector<Segment>& segments, const Configuration& configuration);

private:
	Road road;

	void checkLocalContraints(const Configuration& configuration);

};

#endif