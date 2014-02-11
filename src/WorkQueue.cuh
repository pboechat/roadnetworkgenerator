#ifndef WORKQUEUE_CUH
#define WORKQUEUE_CUH

#pragma once

#include <Constants.h>
#include <CpuGpuCompatibility.h>
#include <MathExtras.h>
#include <cutil.h>

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

	//////////////////////////////////////////////////////////////////////////
	template<typename WorkItemType>
	__device__ void push(WorkItemType& item)
	{
		// FIXME: checking boundaries
		if (count >= MAX_NUM_WORKITEMS)
		{
			THROW_EXCEPTION1("max. number of work items overflow (%d)", count);
		}

		unsigned int mask = __ballot(1);
		unsigned int numberOfActiveThreads = __popc(mask);
		int laneId = __popc(lanemask_lt() & mask);
		int leadingThreadId = __ffs(mask) - 1;

		int firstPushIndex;
		if (laneId == 0)
		{
			atomicAdd((int*)&count, numberOfActiveThreads);

			// FIXME: checking boundaries
			if (count > MAX_NUM_WORKITEMS)
			{
				THROW_EXCEPTION1("max. number of work items overflow (%d)", count);
			}

			firstPushIndex = atomicAdd(&tail, numberOfActiveThreads);
		}

		firstPushIndex = __shfl(firstPushIndex, leadingThreadId);

		unsigned int index = ((firstPushIndex + laneId) % MAX_NUM_WORKITEMS);

		while (readFlags[index]);

		pack(index, item);

		__threadfence();

		readFlags[index] = true;
	}
#else
	//////////////////////////////////////////////////////////////////////////
	template<typename WorkItemType>
	void push(WorkItemType& item)
	{
		// FIXME: identical to push on host

		if (count >= MAX_NUM_WORKITEMS)
		{
			THROW_EXCEPTION1("max. number of work items overflow (%d)", count);
		}

		unsigned int index = tail++ % MAX_NUM_WORKITEMS;

		pack(index, item);
		count++;
	}
#endif

	//////////////////////////////////////////////////////////////////////////
	template<typename WorkItemType>
	void unsafePush(WorkItemType& item)
	{
		if (count >= MAX_NUM_WORKITEMS)
		{
			THROW_EXCEPTION("WorkQueue: count >= MAX_NUM_WORKITEMS");
		}

		unsigned int index = tail++ % MAX_NUM_WORKITEMS;

		pack(index, item);
		count++;

#ifdef USE_CUDA
		readFlags[index] = true;
#endif
	}

	//////////////////////////////////////////////////////////////////////////
	template<typename WorkItemType>
	HOST_CODE void unsafePop(WorkItemType& item)
	{
		// FIXME: checking invariants
		if (count == 0)
		{
			THROW_EXCEPTION("count == 0");
		}
		
		// FIXME: checking invariants
		if (head == tail)
		{
			THROW_EXCEPTION("head == tail");
		}

		unsigned int index = head++ % MAX_NUM_WORKITEMS;
		unpack(index, item);
		count--;

#ifdef USE_CUDA
		readFlags[index] = false;
#endif
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