#ifndef SORTEDSET_H
#define SORTEDSET_H

#include <exception>

template<typename T>
class SortedSet
{
public:
	typedef int (*Comparer)(const T&, const T&);

	SortedSet(T* data, unsigned int capacity, Comparer cmp) : data(data), capacity(capacity), count(0), cmp(cmp) {}
	~SortedSet() {}

	void insert(const T& item)
	{
		if (count >= capacity)
		{
			// FIXME: checking boundaries
			throw std::exception("count >= capacity");
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

	void remove(const T& item)
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

	int indexOf(const T& item) const
	{
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

	inline unsigned int size() const
	{
		return count; 
	}

	inline T& operator[] (unsigned int i)
	{
		return data[i];
	}

	inline const T& operator[] (unsigned int i) const
	{
		return data[i];
	}

private:
	T* data;
	unsigned int capacity;
	unsigned int count;
	Comparer cmp;

	void binarySearch(const T& item, int& min, int& max) const
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