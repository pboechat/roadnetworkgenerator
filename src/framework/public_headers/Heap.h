#ifndef HEAP_H
#define HEAP_H

#include <exception>

template <typename T>
class Heap
{
public:
	typedef int (*Compare)(const T&, const T&);

	Heap(T* data, unsigned int capacity, Compare cmp) : data(data), capacity(capacity), count(0), cmp(cmp) {}
	~Heap() {}

	inline int getLeft(int parent) const
	{
		int i = (parent << 1) + 1; // 2 * parent + 1
		return (i < (int)capacity) ? i : -1;
	}

	inline int getRight(int parent) const
	{
		int i = (parent << 1) + 2; // 2 * parent + 2
		return (i < (int)capacity) ? i : -1;
	}

	inline int getParent(int child) const
	{
		if (child != 0)
		{
			int i = (child - 1) >> 1;
			return i;
		}
		return -1;
	}

	void insert(const T& item)
	{
		if (count >= capacity)
		{
			// FIXME: checking boundaries
			throw std::exception("elements >= size");
		}

		data[count] = item;
		heapifyUp(count++);
	}

	inline T& operator[] (unsigned int i)
	{
		return data[i];
	}

	inline const T& operator[] (unsigned int i) const
	{
		return data[i];
	}

	T& popFirst()
	{
		if (count == 0)
		{
			// FIXME: checking boundaries
			throw std::exception("count == 0");
		}

		T& first = data[0];
		data[0] = data[count - 1];
		count--;
		heapifyDown(0);
		return first;
	}

	void remove(const T& item)
	{
		int i = find(item);

		if (i < 0)
		{
			// FIXME: checking boundaries
			throw std::exception("i < 0");
		}

		remove((unsigned int)i);
	}

	void remove(unsigned int i)
	{
		int left = getLeft(i);
		int successor = -1;
		if (left == -1)
		{
			int right = getRight(i);
			if (right == -1)
			{
				successor = right;
				left = getLeft(right);
				while (left != -1)
				{
					successor = left;
					left = getLeft(right);
				}
			}
		}
		else
		{
			successor = left;
			int right = getRight(left);
			while (right != -1)
			{
				successor = right;
				right = getRight(right);
			}
		}

		if (successor != -1)
		{
			data[i] = data[successor];
		}

		count--;
		heapifyDown(i);
	}

	inline bool empty() const
	{
		return count == 0;
	}

	inline unsigned int size() const
	{
		return count;
	}

private:
	T* data;
	unsigned int capacity;
	unsigned int count;
	Compare cmp;

	int find(const T& item) const
	{
		int index = (count - 1) >> 1;
		int compare;
		while ((compare = cmp(data[index], item)) != 0)
		{
			if (compare < 0)
			{
				index += (index >> 1);
			}
			else
			{
				index -= (index >> 1);
			}
		}

		if (compare == 0)
		{
			return index;
		}
		else
		{
			return -1;
		}
	}

	void heapifyUp(unsigned int index)
	{
		int parent = getParent(index);
		while (index > 0 && parent >= 0 && cmp(data[parent], data[index]) == 1)
		{
			T tmp = data[parent];
			data[parent] = data[index];
			data[index] = tmp;

			index = parent;
			parent = getParent(index);
		}
	}

	void heapifyDown(unsigned int index)
	{
		int left = getLeft(index);
		int right = getRight(index);
		if (left > 0 && right > 0 && cmp(data[left], data[right]) == 1)
		{
			left = right;
		}

		if (left > 0)
		{
			T tmp = data[index];
			data[index] = data[left];
			data[left] = tmp;

			heapifyDown(left);
		}
	}

};

#endif