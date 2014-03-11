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
	//volatile bool readFlags[WORK_QUEUE_DATA_SIZE];

#ifdef USE_CUDA
	//////////////////////////////////////////////////////////////////////////
	__device__ void reservePops(int reserves, unsigned int& first, unsigned int& reserved)
	{
		if (count <= 0)
		{
			reserved = 0;
			return;
		}

		int oldCount = atomicSub((int*)&count, reserves);

		if (oldCount > 0)
		{
			int overflow = MathExtras::min<int>(oldCount - reserves, 0);
			if (overflow < 0)
			{
				atomicSub((int*)&count, overflow);
			}

			reserved = reserves + overflow;
			first = atomicAdd(&head, reserved);
		}
		else
		{
			reserved = 0;
			atomicAdd((int*)&count, reserves);
		}
	}

	//////////////////////////////////////////////////////////////////////////
	template<typename WorkItemType>
	__device__ void popReserved(unsigned int index, WorkItemType& item)
	{
		index = (index % MAX_NUM_WORKITEMS);
		//while (!readFlags[index]);
		unpack(index, item);
		//readFlags[index] = false;
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
		unsigned int activeThreads = __popc(mask);
		int lane = __popc(lanemask_lt() & mask);
		int leadingThread = __ffs(mask) - 1;

		int first;
		if (lane == 0)
		{
			atomicAdd((int*)&count, activeThreads);
			// FIXME: checking boundaries
			if (count > MAX_NUM_WORKITEMS)
			{
				THROW_EXCEPTION1("max. number of work items overflow (%d)", count);
			}
			first = atomicAdd(&tail, activeThreads);
		}
		first = __shfl(first, leadingThread);

		unsigned int index = ((first + lane) % MAX_NUM_WORKITEMS);
		//while (readFlags[index]);
		pack(index, item);
		//readFlags[index] = true;
		__threadfence();
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
		//readFlags[index] = true;
		count++;
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
		//readFlags[index] = false;
		count--;
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