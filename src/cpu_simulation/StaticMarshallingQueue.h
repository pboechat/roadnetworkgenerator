#ifndef STATICMARSHALLINGQUEUE_H
#define STATICMARSHALLINGQUEUE_H

#include <exception>

class StaticMarshallingQueue
{
public:
	StaticMarshallingQueue() : buffer(0), capacity(0), itemSize(0), counter(0), head(0), tail(0)
	{
	}

	void setBuffer(unsigned char* buffer, unsigned int capacity)
	{
		this->buffer = buffer;
		this->capacity = capacity;
	}

	void setItemSize(unsigned int itemSize)
	{
		this->itemSize = itemSize;
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