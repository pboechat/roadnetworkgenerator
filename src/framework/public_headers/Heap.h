#ifndef HEAP_H
#define HEAP_H

#include <exception>

template <typename T>
class Heap
{
public:
	typedef bool (*Comparer)(const T&, const T&);

	Heap(T* data, unsigned int capacity, Comparer compare) : data(data), capacity(capacity), count(0), compare(compare) {}
	~Heap() {}

	inline int left(unsigned int i) const
	{
		return (i * 2) + 1;
	}

	inline int right(unsigned int i) const
	{
		if (i >= (count - 1) / 2)
		{
			// FIXME: checking boundaries
			throw std::exception("i < (elements - 1) / 2");
		}

		return (i * 2) + 2;
	}

	inline int parent(unsigned int i) const
	{
		if (i == 0)
		{
			// FIXME: checking boundaries
			throw std::exception("i == 0");
		}

		return (i - 1) / 2;
	}

	void insert(const T& item)
	{
		if (count >= capacity)
		{
			// FIXME: checking boundaries
			throw std::exception("elements >= size");
		}

		unsigned int i = count++;
		data[i] = item;

		if (i == 0)
		{
			return;
		}

		unsigned int p = parent(i);
		//while (data[i] > data[p])
		while (compare(data[i], data[p]))
		{
			T& tmp = data[i];
			data[i] = data[p];
			data[p] = tmp;
			i = parent(i);
			p = parent(i);
		}
	}

	T& peekFirst()
	{
		return data[0];
	}

	T& popFirst()
	{
		if (count == 0)
		{
			// FIXME: checking boundaries
			throw std::exception("elements == 0");
		}

		count--;

		if (count != 0)
		{
			T& tmp = data[0];
			data[0] = data[count];
			data[count] = tmp;
			heapify(0);
		}

		return data[count];
	}

	void remove(T& item)
	{
		// TODO:
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
	Comparer compare;

	void heapify(unsigned int root)
	{
		unsigned int child = left(root);

		// if a right child exists, and it's bigger than the left child, it will be used
		//if (data[child] < data[child + 1] && child < count - 1)
		if (!compare(data[child], data[child + 1]) && child < count - 1)
		{
			child++;
		}

		// if root is greater than its biggest child, stop
		//if (data[root] >= data[child])
		if (compare(data[root], data[child]))
		{
			return;
		}

		// swap root and its biggest child
		T& tmp = data[root];
		data[root] = data[child];
		data[child] = tmp;

		// continue the process on root's new children
		heapify(child);
	}

};

#endif