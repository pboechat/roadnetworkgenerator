#ifndef INSTANTIATEROAD_H
#define INSTANTIATEROAD_H

#include <Procedure.h>
#include <EvaluateBranch.h>
#include <EvaluateRoad.h>

class InstantiateRoad : public Procedure
{
public:
	virtual unsigned int getCode()
	{
		return 0;
	}

	virtual void execute(WorkQueuesManager<Procedure>& workQueuesManager, std::vector<Segment>& segments)
	{
		workQueuesManager.addWorkItem(new EvaluateBranch());
		workQueuesManager.addWorkItem(new EvaluateBranch());
		workQueuesManager.addWorkItem(new EvaluateRoad());
	}

};

#endif