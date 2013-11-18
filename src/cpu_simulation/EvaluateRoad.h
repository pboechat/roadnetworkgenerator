#ifndef EVALUATEROAD_H
#define EVALUATEROAD_H

#include <Procedure.h>

class EvaluateRoad : public Procedure
{
public:
	virtual unsigned int getCode()
	{
		return 2;
	}

	virtual void execute(WorkQueuesManager<Procedure>& workQueuesManager, std::vector<Segment>& segments)
	{
		// TODO:
	}

};

#endif