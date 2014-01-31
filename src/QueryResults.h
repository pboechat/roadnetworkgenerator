#ifndef QUERYRESULTS_H
#define QUERYRESULTS_H

#include <Constants.h>
#include <CpuGpuCompatibility.h>

struct QueryResults
{
	int results[MAX_RESULTS_PER_QUERY];
	unsigned int numResults;
	volatile int owner;

	HOST_AND_DEVICE_CODE QueryResults() : numResults(0), owner(-1) {}
	HOST_AND_DEVICE_CODE ~QueryResults() {}

};

#endif