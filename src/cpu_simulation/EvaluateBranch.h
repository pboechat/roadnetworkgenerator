#ifndef EVALUATEBRANCH_H
#define EVALUATEBRANCH_H

#include <Procedure.h>
#include <Branch.h>

class EvaluateBranch : public Procedure
{
public:
	EvaluateBranch(const Branch& branch);

	virtual unsigned int getCode();
	virtual void execute(WorkQueuesManager<Procedure>& workQueuesManager, std::vector<Segment>& segments, ImageMap& populationDensityMap, ImageMap& waterBodiesMap);

private:
	Branch branch;

};

#endif