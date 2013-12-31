#ifndef PROCEDURESDECLARATIONS_H
#define PROCEDURESDECLARATIONS_H

#include <WorkQueuesSet.h>
#include <Road.h>
#include <Branch.h>

#include <vector_math.h>

#include <exception>

//////////////////////////////////////////////////////////////////////////
// PROCEDURES DECLARATIONS
//////////////////////////////////////////////////////////////////////////

#define PROCEDURE_DECL(ProcName, ArgType) \
	struct ProcName \
	{ \
		static void execute(ArgType& item, WorkQueuesSet* backQueues); \
	}

PROCEDURE_DECL(EvaluateBranch, Branch);
PROCEDURE_DECL(EvaluateRoad, Road);
PROCEDURE_DECL(InstantiateRoad, Road);

#endif