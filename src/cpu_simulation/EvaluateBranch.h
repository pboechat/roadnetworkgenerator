#ifndef EVALUATEBRANCH_H
#define EVALUATEBRANCH_H

#include <Procedure.h>

class EvaluateBranch : public Procedure
{
public:
	virtual unsigned int getCode()
	{
		return 1;
	}

	virtual void execute(WorkQueuesManager<Procedure>& workQueuesManager, std::vector<Segment>& segments)
	{
		// TODO:
	}

};

#endif