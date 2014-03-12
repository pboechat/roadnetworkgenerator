#ifndef EXPANSION_KERNEL_CUH
#define EXPANSION_KERNEL_CUH

#include <Constants.h>
#include <CpuGpuCompatibility.h>
#include <Procedures.h>
#include <WorkQueue.cuh>

#ifdef USE_CUDA
//////////////////////////////////////////////////////////////////////////
__device__ volatile unsigned int g_dCounter;

//////////////////////////////////////////////////////////////////////////
__global__ void initializeExpansionKernel()
{
	g_dCounter = 0;
}

//////////////////////////////////////////////////////////////////////////
__global__ void expansionKernel(unsigned int numDerivations, WorkQueue* workQueues1, WorkQueue* workQueues2, unsigned int startingQueue, unsigned int numQueues, Context* context)
{
	__shared__ WorkQueue* frontQueues;
	__shared__ WorkQueue* backQueues;
	__shared__ unsigned int reserved;
	__shared__ unsigned int first;
	__shared__ unsigned int derivation;
	__shared__ unsigned int currentQueue;
	__shared__ unsigned int state;

	if (threadIdx.x == 0)
	{
		frontQueues = workQueues1;
		backQueues = workQueues2;
		derivation = 0;
		state = 0;
		currentQueue = startingQueue + (blockIdx.x % numQueues);
	}

	__syncthreads();

	while (derivation < numDerivations)
	{
		if (state == 0)
		{
			if (threadIdx.x == 0)
			{
				state = 1;
				atomicAdd((int*)&g_dCounter, 1);
			}

			__syncthreads();

			while (state == 1)
			{
				if (threadIdx.x == 0)
				{
					frontQueues[currentQueue].reservePops(blockDim.x, first, reserved);
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

				if (threadIdx.x == 0 && reserved == 0)
				{
					state = 2;
					atomicSub((int*)&g_dCounter, 1);
				}

				__syncthreads();
			}
		}

		if (threadIdx.x == 0 && g_dCounter == 0)
		{
			derivation++;
			WorkQueue* tmp = frontQueues;
			frontQueues = backQueues;
			backQueues = tmp;
			state = 0;
		}

		__syncthreads();
	}
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