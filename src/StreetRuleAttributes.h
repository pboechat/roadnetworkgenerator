#ifndef STREETRULEATTRIBUTES_H
#define STREETRULEATTRIBUTES_H

#pragma once

#include <CpuGpuCompatibility.h>

#define EXPAND_UP		1
#define EXPAND_RIGHT	2
#define EXPAND_LEFT		4

struct StreetRuleAttributes
{
	unsigned int branchDepth;
	unsigned int boundsIndex;
	unsigned char expansionMask;

	HOST_AND_DEVICE_CODE StreetRuleAttributes() : branchDepth(0), boundsIndex(0), expansionMask(EXPAND_UP | EXPAND_RIGHT) {}
	HOST_AND_DEVICE_CODE StreetRuleAttributes(unsigned int branchDepth, unsigned int boundsIndex) : branchDepth(branchDepth), boundsIndex(boundsIndex), expansionMask(EXPAND_UP | EXPAND_RIGHT) {}
	HOST_AND_DEVICE_CODE ~StreetRuleAttributes() {}

	HOST_AND_DEVICE_CODE StreetRuleAttributes& operator = (const StreetRuleAttributes& other)
	{
		branchDepth = other.branchDepth;
		boundsIndex = other.boundsIndex;
		expansionMask = other.expansionMask;
		return *this;
	}

};

#endif