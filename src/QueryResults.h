#ifndef QUERYRESULTS_H
#define QUERYRESULTS_H

#include <Constants.h>
#include <CpuGpuCompatibility.h>

struct QueryResults
{
	int results[MAX_RESULTS_PER_QUERY];
	unsigned int numResults;

	HOST_AND_DEVICE_CODE QueryResults() : numResults(0) {}
	HOST_AND_DEVICE_CODE ~QueryResults() {}

};

#endif