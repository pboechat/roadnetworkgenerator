#ifndef STREETRULEATTRIBUTES_H
#define STREETRULEATTRIBUTES_H

#include "Defines.h"

HOST_AND_DEVICE_CODE struct StreetRuleAttributes
{
	unsigned int branchDepth;

	StreetRuleAttributes() : branchDepth(0) {}

};

#endif