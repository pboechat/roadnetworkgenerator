#ifndef STREETRULEATTRIBUTES_CUH
#define STREETRULEATTRIBUTES_CUH

#include "Defines.h"

struct StreetRuleAttributes
{
	unsigned int branchDepth;

	HOST_AND_DEVICE_CODE StreetRuleAttributes() : branchDepth(0) {}
	HOST_AND_DEVICE_CODE ~StreetRuleAttributes() {}

	HOST_AND_DEVICE_CODE StreetRuleAttributes& operator = (const StreetRuleAttributes& other)
	{
		branchDepth = other.branchDepth;
		return *this;
	}

};

#endif