#ifndef EXPANSION_KERNEL_CUH
#define EXPANSION_KERNEL_CUH

#include <Constants.h>
#include <CpuGpuCompatibility.h>
#include <Procedures.h>
#include <WorkQueue.cuh>

#ifdef USE_CUDA
//////////////////////////////////////////////////////////////////////////
__global__ void expansionKernel(WorkQueue* frontQueues, WorkQueue* backQueues, unsigned int startingQueue, unsigned int numQueues, Context* context)
{
	__shared__ unsigned int reserved;
	__shared__ unsigned int first;
	__shared__ unsigned int currentQueue;
	__shared__ bool runAgain;

	do
	{
		if (threadIdx.x == 0)
		{
			currentQueue = startingQueue + (blockIdx.x % numQueues);
			frontQueues[currentQueue].reservePops(blockDim.x, first, reserved);
			runAgain = (reserved > 0);
		}

		__syncthreads();

		if (threadIdx.x < reserved)
		{
			unsigned int index = first + threadIdx.x;
			switch (currentQueue)
			{
			case EVALUATE_HIGHWAY_BRANCH:
				{
					HighwayBranch highwayBranch;
					frontQueues[EVALUATE_HIGHWAY_BRANCH].popReserved(index, highwayBranch);
					EvaluateHighwayBranch::execute(highwayBranch, context, backQueues);
				}
				break;
			case EVALUATE_HIGHWAY:
				{
					Highway highway;
					frontQueues[EVALUATE_HIGHWAY].popReserved(index, highway);
					EvaluateHighway::execute(highway, context, backQueues);
				}
				break;
			case INSTANTIATE_HIGHWAY:
				{
					Highway highway;
					frontQueues[INSTANTIATE_HIGHWAY].popReserved(index, highway);
					InstantiateHighway::execute(highway, context, backQueues);
				}
				break;
			case EVALUATE_STREET:
				{
					Street street;
					frontQueues[EVALUATE_STREET].popReserved(index, street);
					EvaluateStreet::execute(street, context, backQueues);
				}
				break;
			case INSTANTIATE_STREET:
				{
					Street street;
					frontQueues[INSTANTIATE_STREET].popReserved(index, street);
					InstantiateStreet::execute(street, context, backQueues);
				}
				break;
			default:
				THROW_EXCEPTION("invalid queue index");
			}
		}
	} while (runAgain);
}
#else
//////////////////////////////////////////////////////////////////////////
void expansionKernel(unsigned int numDerivations, WorkQueue* workQueues1, WorkQueue* workQueues2, unsigned int startingQueue, unsigned int numQueues, Context* context)
{
	WorkQueue* frontQueues = workQueues1;
	WorkQueue* backQueues = workQueues2;

	for (unsigned int i = 0; i < numDerivations; i++)
	{
		for (unsigned int j = 0; j < numQueues; j++)
		{
			unsigned int currentQueue = startingQueue + j;
			switch (currentQueue)
			{
			case EVALUATE_HIGHWAY_BRANCH:
				{
					HighwayBranch highwayBranch;
					while (frontQueues[EVALUATE_HIGHWAY_BRANCH].count > 0)
					{
						frontQueues[EVALUATE_HIGHWAY_BRANCH].unsafePop(highwayBranch);
						EvaluateHighwayBranch::execute(highwayBranch, context, backQueues);
					}
				}
				break;
			case EVALUATE_HIGHWAY:
				{
					Highway highway;
					while (frontQueues[EVALUATE_HIGHWAY].count > 0)
					{
						frontQueues[EVALUATE_HIGHWAY].unsafePop(highway);
						EvaluateHighway::execute(highway, context, backQueues);
					}
				}
				break;
			case INSTANTIATE_HIGHWAY:
				{
					Highway highway;
					while (frontQueues[INSTANTIATE_HIGHWAY].count > 0)
					{
						frontQueues[INSTANTIATE_HIGHWAY].unsafePop(highway);
						InstantiateHighway::execute(highway, context, backQueues);
					}
				}
				break;
			case EVALUATE_STREET:
				{
					Street street;
					while (frontQueues[EVALUATE_STREET].count > 0)
					{
						frontQueues[EVALUATE_STREET].unsafePop(street);
						EvaluateStreet::execute(street, context, backQueues);
					}
				}
				break;
			case INSTANTIATE_STREET:
				{
					Street street;
					while (frontQueues[INSTANTIATE_STREET].count > 0)
					{
						frontQueues[INSTANTIATE_STREET].unsafePop(street);
						InstantiateStreet::execute(street, context, backQueues);
					}
				}
				break;
			default:
				THROW_EXCEPTION("invalid queue index");
			}
		}
		WorkQueue* tmp = frontQueues;
		frontQueues = backQueues;
		backQueues = tmp;
	}
}
#endif

#endif