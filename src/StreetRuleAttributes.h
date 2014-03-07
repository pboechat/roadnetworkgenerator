#ifndef STREETRULEATTRIBUTES_H
#define STREETRULEATTRIBUTES_H

#pragma once

#include <CpuGpuCompatibility.h>

#define RIGHT_CHILD 1
#define LEFT_CHILD 2
#define UP_CHILD 3
#define DOWN_CHILD 4

struct StreetRuleAttributes
{
	unsigned int branchDepth;
	unsigned int boundsIndex;
	char childCode;

	HOST_AND_DEVICE_CODE StreetRuleAttributes() : branchDepth(0), boundsIndex(0), childCode(0) {}
	HOST_AND_DEVICE_CODE StreetRuleAttributes(unsigned int branchDepth, unsigned int boundsIndex, char childCode) : branchDepth(branchDepth), boundsIndex(boundsIndex), childCode(childCode) {}
	HOST_AND_DEVICE_CODE ~StreetRuleAttributes() {}

	HOST_AND_DEVICE_CODE StreetRuleAttributes& operator = (const StreetRuleAttributes& other)
	{
		branchDepth = other.branchDepth;
		boundsIndex = other.boundsIndex;
		childCode = other.childCode;
		return *this;
	}

};

#endif