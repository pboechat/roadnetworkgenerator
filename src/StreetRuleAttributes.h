#ifndef STREETRULEATTRIBUTES_H
#define STREETRULEATTRIBUTES_H

#pragma once

#include <CpuGpuCompatibility.h>

struct StreetRuleAttributes
{
	unsigned int branchDepth;
	unsigned int boundsIndex;

	HOST_AND_DEVICE_CODE StreetRuleAttributes() : branchDepth(0), boundsIndex(0) {}
	HOST_AND_DEVICE_CODE StreetRuleAttributes(unsigned int branchDepth, unsigned int boundsIndex) : branchDepth(branchDepth), boundsIndex(boundsIndex) {}
	HOST_AND_DEVICE_CODE ~StreetRuleAttributes() {}

	HOST_AND_DEVICE_CODE StreetRuleAttributes& operator = (const StreetRuleAttributes& other)
	{
		branchDepth = other.branchDepth;
		boundsIndex = other.boundsIndex;
		return *this;
	}

};

#endif