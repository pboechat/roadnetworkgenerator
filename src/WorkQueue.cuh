#ifndef WORKQUEUE_CUH
#define WORKQUEUE_CUH

#include "Defines.h"
#include <MathExtras.cuh>

#ifdef USE_CUDA
//////////////////////////////////////////////////////////////////////////
inline __device__ unsigned int lanemask_lt()
{
	unsigned int lanemask;
	asm("mov.u32 %0, %lanemask_lt;" : "=r" (lanemask));
	return lanemask;
}
#endif

struct WorkQueue
{
	unsigned int head;
	unsigned int tail;
	volatile int count;
	unsigned char data[WORK_QUEUE_DATA_SIZE];
#ifdef USE_CUDA
	volatile bool readFlags[MAX_NUM_WORKITEMS];
#endif

#ifdef USE_CUDA
	//////////////////////////////////////////////////////////////////////////
	__device__ void reservePops(unsigned int size, unsigned int* firstPopIndex, unsigned int* reservedPops)
	{
		if (count <= 0)
		{
			*reservedPops = 0;
			return;
		}

		int oldCount = atomicSub((int*)&count, (int)size);

		if (oldCount > 0)
		{
			int overflow = MathExtras::min<int>(oldCount - (int)size, 0);
			if (overflow < 0)
			{
				atomicSub((int*)&count, overflow);
			}

			*reservedPops = size + overflow;
			*firstPopIndex = atomicAdd(&head, *reservedPops);
		}
		else
		{
			*reservedPops = 0;
			atomicAdd((int*)&count, size);
		}
	}

	//////////////////////////////////////////////////////////////////////////
	template<typename WorkItemType>
	__device__ void popReserved(unsigned int index, WorkItemType& item)
	{
		index = (index % MAX_NUM_WORKITEMS);

		while (!readFlags[index]);

		unpack(index, item);

		__threadfence();

		readFlags[index] = false;
	}

	#if CUDA_CC >= 30
		//////////////////////////////////////////////////////////////////////////
		template<typename WorkItemType>
		__device__ bool push(WorkItemType& item)
		{
			if (count >= MAX_NUM_WORKITEMS)
			{
				THROW_EXCEPTION("WorkQueue: count >= MAX_NUM_WORKITEMS");
			}

			unsigned int mask = __ballot(1);
			unsigned int numberOfActiveThreads = __popc(mask);
			int laneId = __popc(lanemask_lt() & mask);
			int leadingThreadId = __ffs(mask) - 1;

			int firstPushIndex;
			int pushes = 0;
			if (laneId == 0)
			{
				int oldCount = atomicAdd((int*)&count, numberOfActiveThreads);

				if (oldCount < MAX_NUM_WORKITEMS)
				{
					int overflow = MathExtras::min(MAX_NUM_WORKITEMS - (int)numberOfActiveThreads - oldCount, 0);
					if (overflow < 0)
					{
						atomicAdd((int*)&count, overflow);
					}

					pushes = numberOfActiveThreads + overflow;
					firstPushIndex = atomicAdd(&tail, pushes);
				}
			}

			firstPushIndex = __shfl(firstPushIndex, leadingThreadId);
			pushes = __shfl(pushes, leadingThreadId);

			if (laneId < pushes)
			{
				unsigned int index = ((firstPushIndex + laneId) % MAX_NUM_WORKITEMS);

				while (readFlags[index]);

				pack(index, item);

				__threadfence();

				readFlags[index] = true;

				return true;
			}

			return false;
		}
	#else
		template<typename WorkItemType>
		__device__ bool push(WorkItemType& item)
		{
			if (count >= MAX_NUM_WORKITEMS)
			{
				THROW_EXCEPTION("WorkQueue: count >= MAX_NUM_WORKITEMS");
			}

			int oldCount = atomicAdd((int*)&count, 1);

			if (oldCount >= MAX_NUM_WORKITEMS)
			{
				atomicSub((int*)&count, 1);
				return false;
			}

			unsigned int index = atomicAdd(&tail, 1);

			while (readFlags[index]);

			pack(index, item);

			__threadfence();

			readFlags[index] = true;

			return true;
		}
	#endif
#endif

	//////////////////////////////////////////////////////////////////////////
	template<typename WorkItemType>
	HOST_CODE void pushOnHost(WorkItemType& item)
	{
		if (count >= MAX_NUM_WORKITEMS)
		{
			THROW_EXCEPTION("WorkQueue: count >= MAX_NUM_WORKITEMS");
		}

		unsigned int index = tail;
		tail = ++tail % MAX_NUM_WORKITEMS;

		if (tail == head)
		{
			THROW_EXCEPTION("WorkQueue: tail == head");
		}

		pack(index, item);
		count++;
	}

	//////////////////////////////////////////////////////////////////////////
	template<typename WorkItemType>
	HOST_CODE void popOnHost(WorkItemType& item)
	{
		if (count == 0)
		{
			THROW_EXCEPTION("WorkQueue: count == 0");
		}
		
		if (head == tail)
		{
			THROW_EXCEPTION("WorkQueue: head == tail");
		}

		unsigned int index = head;
		head = ++head % MAX_NUM_WORKITEMS;
		unpack(index, item);
		count--;
	}

	//////////////////////////////////////////////////////////////////////////
	HOST_CODE void clear()
	{
		head = tail = count = 0;
	}

private:
	//////////////////////////////////////////////////////////////////////////
	template<typename WorkItemType>
	HOST_AND_DEVICE_CODE void pack(unsigned int index, WorkItemType& item)
	{
		unsigned int offset = index * WORK_ITEM_SIZE;
		*reinterpret_cast<WorkItemType*>(data + offset) = item;
	}

	//////////////////////////////////////////////////////////////////////////
	template<typename WorkItemType>
	HOST_AND_DEVICE_CODE void unpack(unsigned int index, WorkItemType& item)
	{
		unsigned int offset = index * WORK_ITEM_SIZE;
		item = *reinterpret_cast<WorkItemType*>(data + offset);
	}

};


#endif