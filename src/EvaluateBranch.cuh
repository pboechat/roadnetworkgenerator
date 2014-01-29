#ifndef EVALUATEBRANCH_CUH
#define EVALUATEBRANCH_CUH

#include "Defines.h"
#include <WorkQueue.cuh>
#include <Road.cuh>
#include <Branch.cuh>
#include <ProceduresCodes.h>

//////////////////////////////////////////////////////////////////////////
struct EvaluateStreetBranch
{
	//////////////////////////////////////////////////////////////////////////
	static DEVICE_CODE void execute(StreetBranch& branch, WorkQueue* backQueues)
	{
		// p6
		if (branch.delay < 0)
		{
			return;
		}

		// p4
		else if (branch.delay > 0)
		{
			branch.delay--;
			backQueues[EVALUATE_STREET_BRANCH].push(branch);
		}

		// p5
		else if (branch.delay == 0)
		{
			backQueues[EVALUATE_STREET].push(Street(0, branch.roadAttributes, branch.ruleAttributes, UNASSIGNED));
		}
	}
};

//////////////////////////////////////////////////////////////////////////
struct EvaluateHighwayBranch
{
	//////////////////////////////////////////////////////////////////////////
	static DEVICE_CODE void execute(HighwayBranch& branch, WorkQueue* backQueues)
	{
		// p6
		if (branch.delay < 0)
		{
			return;
		}

		// p4
		else if (branch.delay > 0)
		{
			branch.delay--;
			backQueues[EVALUATE_HIGHWAY_BRANCH].push(branch);
		}

		// p5
		else if (branch.delay == 0)
		{
			backQueues[EVALUATE_HIGHWAY].push(Highway(0, branch.roadAttributes, branch.ruleAttributes, UNASSIGNED));
		}
	}
};

#endif