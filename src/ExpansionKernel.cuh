#ifndef EXPANSION_KERNEL_CUH
#define EXPANSION_KERNEL_CUH

#include <Constants.h>
#include <CpuGpuCompatibility.h>
#include <Procedures.h>
#include <WorkQueue.cuh>

#ifdef PARALLEL
#ifdef PERSISTENT_THREADS
//////////////////////////////////////////////////////////////////////////
__device__ volatile unsigned int g_dCounter;

//////////////////////////////////////////////////////////////////////////
__global__ void initializeCounters()
{
	g_dCounter = 0;
}

//////////////////////////////////////////////////////////////////////////
__global__ void expansionKernel(unsigned int numDerivations, WorkQueue* queues1, WorkQueue* queues2, unsigned int startingQueue, unsigned int numQueues, Context* context)
{
	__shared__ WorkQueue* front;
	__shared__ WorkQueue* back;
	__shared__ unsigned int reserved;
	__shared__ unsigned int first;
	__shared__ unsigned int derivation;
	__shared__ unsigned int i;
	__shared__ unsigned int state;

	if (threadIdx.x == 0)
	{
		front = queues1;
		back = queues2;
		derivation = 0;
		state = 0;
		i = startingQueue + (blockIdx.x % numQueues);
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
					front[i].reservePops(blockDim.x, first, reserved);
				}

				__syncthreads();

				if (threadIdx.x < reserved)
				{
					unsigned int index = first + threadIdx.x;
					switch (i)
					{
					case EVALUATE_HIGHWAY_BRANCH:
						{
							HighwayBranch highwayBranch;
							front[EVALUATE_HIGHWAY_BRANCH].popReserved(index, highwayBranch);
							EvaluateHighwayBranch::execute(highwayBranch, context, back);
						}
						break;
					case EVALUATE_HIGHWAY:
						{
							Highway highway;
							front[EVALUATE_HIGHWAY].popReserved(index, highway);
							EvaluateHighway::execute(highway, context, back);
						}
						break;
					case INSTANTIATE_HIGHWAY:
						{
							Highway highway;
							front[INSTANTIATE_HIGHWAY].popReserved(index, highway);
							InstantiateHighway::execute(highway, context, back);
						}
						break;
					case EVALUATE_STREET:
						{
							Street street;
							front[EVALUATE_STREET].popReserved(index, street);
							EvaluateStreet::execute(street, context, back);
						}
						break;
					case INSTANTIATE_STREET:
						{
							Street street;
							front[INSTANTIATE_STREET].popReserved(index, street);
							InstantiateStreet::execute(street, context, back);
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
			WorkQueue* tmp = front;
			front = back;
			back = tmp;
			state = 0;
		}

		__syncthreads();
	}
}
#else
//////////////////////////////////////////////////////////////////////////
__global__ void expansionKernel(WorkQueue* front, WorkQueue* back, unsigned int startingQueue, unsigned int numQueues, Context* context)
{
	__shared__ unsigned int reserved;
	__shared__ unsigned int first;
	__shared__ unsigned int i;
	__shared__ bool runAgain;

	do
	{
		if (threadIdx.x == 0)
		{
			i = startingQueue + (blockIdx.x % numQueues);
			front[i].reservePops(blockDim.x, first, reserved);
			runAgain = (reserved > 0);
		}

		__syncthreads();

		if (threadIdx.x < reserved)
		{
			unsigned int index = first + threadIdx.x;
			switch (i)
			{
			case EVALUATE_HIGHWAY_BRANCH:
				{
					HighwayBranch highwayBranch;
					front[EVALUATE_HIGHWAY_BRANCH].popReserved(index, highwayBranch);
					EvaluateHighwayBranch::execute(highwayBranch, context, back);
				}
				break;
			case EVALUATE_HIGHWAY:
				{
					Highway highway;
					front[EVALUATE_HIGHWAY].popReserved(index, highway);
					EvaluateHighway::execute(highway, context, back);
				}
				break;
			case INSTANTIATE_HIGHWAY:
				{
					Highway highway;
					front[INSTANTIATE_HIGHWAY].popReserved(index, highway);
					InstantiateHighway::execute(highway, context, back);
				}
				break;
			case EVALUATE_STREET:
				{
					Street street;
					front[EVALUATE_STREET].popReserved(index, street);
					EvaluateStreet::execute(street, context, back);
				}
				break;
			case INSTANTIATE_STREET:
				{
					Street street;
					front[INSTANTIATE_STREET].popReserved(index, street);
					InstantiateStreet::execute(street, context, back);
				}
				break;
			default:
				THROW_EXCEPTION("invalid queue index");
			}
		}
	} while (runAgain);
}
#endif
#else
//////////////////////////////////////////////////////////////////////////
void expansionKernel(unsigned int numDerivations, WorkQueue* queues1, WorkQueue* queues2, unsigned int startingQueue, unsigned int numQueues, Context* context)
{
	WorkQueue* front = queues1;
	WorkQueue* back = queues2;

	for (unsigned int i = 0; i < numDerivations; i++)
	{
		for (unsigned int j = 0; j < numQueues; j++)
		{
			unsigned int i = startingQueue + j;
			switch (i)
			{
			case EVALUATE_HIGHWAY_BRANCH:
				{
					HighwayBranch highwayBranch;
					while (front[EVALUATE_HIGHWAY_BRANCH].count > 0)
					{
						front[EVALUATE_HIGHWAY_BRANCH].unsafePop(highwayBranch);
						EvaluateHighwayBranch::execute(highwayBranch, context, back);
					}
				}
				break;
			case EVALUATE_HIGHWAY:
				{
					Highway highway;
					while (front[EVALUATE_HIGHWAY].count > 0)
					{
						front[EVALUATE_HIGHWAY].unsafePop(highway);
						EvaluateHighway::execute(highway, context, back);
					}
				}
				break;
			case INSTANTIATE_HIGHWAY:
				{
					Highway highway;
					while (front[INSTANTIATE_HIGHWAY].count > 0)
					{
						front[INSTANTIATE_HIGHWAY].unsafePop(highway);
						InstantiateHighway::execute(highway, context, back);
					}
				}
				break;
			case EVALUATE_STREET:
				{
					Street street;
					while (front[EVALUATE_STREET].count > 0)
					{
						front[EVALUATE_STREET].unsafePop(street);
						EvaluateStreet::execute(street, context, back);
					}
				}
				break;
			case INSTANTIATE_STREET:
				{
					Street street;
					while (front[INSTANTIATE_STREET].count > 0)
					{
						front[INSTANTIATE_STREET].unsafePop(street);
						InstantiateStreet::execute(street, context, back);
					}
				}
				break;
			default:
				THROW_EXCEPTION("invalid queue index");
			}
		}
		WorkQueue* tmp = front;
		front = back;
		back = tmp;
	}
}
#endif

#endif