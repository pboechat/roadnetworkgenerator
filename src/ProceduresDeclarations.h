#ifndef PROCEDURESDECLARATIONS_H
#define PROCEDURESDECLARATIONS_H

#include "Defines.h"
#include <WorkQueuesSet.h>
#include <Road.h>
#include <Branch.h>

#include <vector_math.h>

#include <exception>

//////////////////////////////////////////////////////////////////////////
// PROCEDURES DECLARATIONS
//////////////////////////////////////////////////////////////////////////

#define PROCEDURE_DECL(ProcName, ArgType) \
	DEVICE_CODE struct ProcName \
	{ \
		static DEVICE_CODE void execute(ArgType& item, WorkQueuesSet* backQueues); \
	}

PROCEDURE_DECL(EvaluateHighwayBranch, HighwayBranch);
PROCEDURE_DECL(EvaluateHighway, Highway);
PROCEDURE_DECL(InstantiateHighway, Highway);
PROCEDURE_DECL(EvaluateStreetBranch, StreetBranch);
PROCEDURE_DECL(EvaluateStreet, Street);
PROCEDURE_DECL(InstantiateStreet, Street);

#endif