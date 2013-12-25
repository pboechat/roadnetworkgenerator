#ifndef STATIC_ALLOC_QUEUE_H
#define STATIC_ALLOC_QUEUE_H

#include <exception>

class static_alloc_queue
{
public:
	static_alloc_queue(unsigned int capacity, unsigned int itemSize) : capacity(capacity), itemSize(itemSize), counter(0), head(0), tail(0), buffer(0)
	{
		buffer = new unsigned char[capacity * itemSize];
	}

	~static_alloc_queue()
	{
		if (buffer != 0)
		{
			delete[] buffer;
		}
	}

	template<typename T>
	T& operator[] (unsigned int i)
	{
		return getItem<T>();
	}

	template<typename T>
	const T& operator[] (unsigned int i) const
	{
		return getItem<T>();
	}

	inline unsigned int size() const
	{
		return counter;
	}

	inline unsigned int getCapacity() const
	{
		return capacity;
	}

	inline unsigned int getItemSize() const
	{
		return itemSize;
	}

	template<typename T>
	void enqueue(const T& item)
	{
		// FIXME: checking invariants
		if (counter >= capacity)
		{
			throw std::exception("counter >= capacity");
		}

		addItem(tail++ % capacity, item);
		counter++;
	}

	template<typename T>
	bool dequeue(T& item)
	{
		if (counter == 0)
		{
			return false;
		}

		getItem(head++ % capacity, item);
		counter--;
		return true;
	}

private:
	unsigned int capacity;
	unsigned int itemSize;
	unsigned int counter;
	unsigned int head;
	unsigned int tail;
	unsigned char* buffer;

	template<typename T>
	void getItem(unsigned int i, T& item) const
	{
		item = *(T*)(buffer + (i * itemSize));
	}

	template<typename T>
	void addItem (unsigned int i, const T& item)
	{
		const unsigned char* data = (const unsigned char*)&item;
		memcpy(buffer + (i * itemSize), data, itemSize);
	}

};

#endif