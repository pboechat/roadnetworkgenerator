#ifndef PROCEDURES_H
#define PROCEDURES_H

#include <Globals.h>
#include <WorkQueues.h>
#include <Road.h>
#include <Branch.h>

#include <vector_math.h>

#include <exception>

//////////////////////////////////////////////////////////////////////////
// PROCEDURE CODES
//////////////////////////////////////////////////////////////////////////

#define EVALUATE_BRANCH 0
#define EVALUATE_ROAD 1
#define INSTANTIATE_ROAD 2

#define NUM_PROCEDURES 3

//////////////////////////////////////////////////////////////////////////
// PROCEDURE DECLARATIONS
//////////////////////////////////////////////////////////////////////////

#define PROCEDURE_DECL(ProcName, ArgType) \
	struct ProcName \
	{ \
		static void execute(ArgType& item, WorkQueues* backQueues); \
	}

PROCEDURE_DECL(EvaluateBranch, Branch);
PROCEDURE_DECL(EvaluateRoad, Road);
PROCEDURE_DECL(InstantiateRoad, Road);

#endif