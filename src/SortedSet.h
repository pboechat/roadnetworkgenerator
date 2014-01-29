#ifndef SORTEDSET_H
#define SORTEDSET_H

#include "Defines.h"

template<typename T>
class SortedSet
{
public:
	struct Comparer
	{
		virtual int operator()(const T&, const T&) const = 0;

	};

	HOST_CODE SortedSet(T* data, unsigned int capacity, const Comparer& cmp) : data(data), capacity(capacity), count(0), cmp(cmp) {}
	HOST_CODE ~SortedSet() {}

	HOST_CODE void insert(const T& item)
	{
		if (count >= capacity)
		{
			// FIXME: checking boundaries
			THROW_EXCEPTION("SortedSet: count >= capacity");
		}

		if (count == 0)
		{
			data[count++] = item;
		}

		else
		{
			int min;
			int max;
			binarySearch(item, min, max);
			T curr = item;
			count++;

			for (; min < (int)count; min++)
			{
				T prev = data[min];
				data[min] = curr;
				curr = prev;
			}
		}
	}

	HOST_CODE void remove(const T& item)
	{
		unsigned int i = indexOf(item);

		if (i == -1)
		{
			return;
		}

		for (; i < count - 1; i++)
		{
			data[i] = data[i + 1];
		}

		count--;
	}

	HOST_CODE int indexOf(const T& item) const
	{
		if (count == 0)
		{
			return -1;
		}

		int min;
		int max;
		binarySearch(item, min, max);

		if (max == min && cmp(data[min], item) == 0)
		{
			return min;
		}

		else
		{
			return -1;
		}
	}

	inline HOST_CODE unsigned int size() const
	{
		return count;
	}

	inline HOST_CODE T& operator[] (unsigned int i)
	{
		return data[i];
	}

	inline HOST_CODE const T& operator[] (unsigned int i) const
	{
		return data[i];
	}

private:
	T* data;
	unsigned int capacity;
	unsigned int count;
	const Comparer& cmp;

	HOST_CODE void binarySearch(const T& item, int& min, int& max) const
	{
		min = 0;
		max = count;

		while (min < max)
		{
			int mid = (min + max) >> 1;

			if (cmp(data[mid], item) < 0)
			{
				min = mid + 1;
			}

			else
			{
				max = mid;
			}
		}
	}

};

#endif