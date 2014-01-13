#ifndef PROCEDURESDECLARATIONS_H
#define PROCEDURESDECLARATIONS_H

#include <WorkQueuesSet.h>
#include <Road.h>
#include <Branch.h>
#include <HighwayRuleAttributes.h>
#include <StreetRuleAttributes.h>

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

PROCEDURE_DECL(EvaluateHighwayBranch, Branch<HighwayRuleAttributes>);
PROCEDURE_DECL(EvaluateHighway, Road<HighwayRuleAttributes>);
PROCEDURE_DECL(InstantiateHighway, Road<HighwayRuleAttributes>);
PROCEDURE_DECL(EvaluateStreetBranch, Branch<StreetRuleAttributes>);
PROCEDURE_DECL(EvaluateStreet, Road<StreetRuleAttributes>);
PROCEDURE_DECL(InstantiateStreet, Road<StreetRuleAttributes>);

#endif