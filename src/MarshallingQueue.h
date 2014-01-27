#ifndef MARSHALLINGQUEUE_H
#define MARSHALLINGQUEUE_H

#include "Defines.h"
#include <exception>

HOST_AND_DEVICE_CODE class MarshallingQueue
{
public:
	HOST_AND_DEVICE_CODE MarshallingQueue() : buffer(0), capacity(0), itemSize(0), counter(0), head(0), tail(0) {}
	HOST_AND_DEVICE_CODE ~MarshallingQueue() {}

	inline HOST_AND_DEVICE_CODE void setBuffer(unsigned char* buffer, unsigned int capacity)
	{
		this->buffer = buffer;
		this->capacity = capacity;
	}

	inline HOST_AND_DEVICE_CODE void setItemSize(unsigned int itemSize)
	{
		this->itemSize = itemSize;
	}

	template<typename T>
	HOST_AND_DEVICE_CODE T& operator[] (unsigned int i)
	{
		return getItem<T>();
	}

	template<typename T>
	HOST_AND_DEVICE_CODE const T& operator[] (unsigned int i) const
	{
		return getItem<T>();
	}

	inline HOST_AND_DEVICE_CODE unsigned int size() const
	{
		return counter;
	}

	inline HOST_AND_DEVICE_CODE unsigned int getCapacity() const
	{
		return capacity;
	}

	inline HOST_AND_DEVICE_CODE unsigned int getItemSize() const
	{
		return itemSize;
	}

	template<typename T>
	HOST_AND_DEVICE_CODE void enqueue(const T& item)
	{
		// FIXME: checking invariants
		if (counter >= capacity)
		{
			//throw std::exception("StaticMarshallingQueue: counter >= capacity");
			THROW_EXCEPTION("StaticMarshallingQueue: counter >= capacity");
		}

		addItem(tail++ % capacity, item);
		counter++;
	}

	template<typename T>
	HOST_AND_DEVICE_CODE bool dequeue(T& item)
	{
		if (counter == 0)
		{
			return false;
		}

		getItem(head++ % capacity, item);
		counter--;
		return true;
	}

	inline HOST_AND_DEVICE_CODE void clear()
	{
		counter = 0;
		head = 0;
		tail = 0;
	}

private:
	unsigned int capacity;
	unsigned int itemSize;
	unsigned int counter;
	unsigned int head;
	unsigned int tail;
	unsigned char* buffer;

	template<typename T>
	HOST_AND_DEVICE_CODE void getItem(unsigned int i, T& item) const
	{
		item = *(T*)(buffer + (i * itemSize));
	}

	template<typename T>
	HOST_AND_DEVICE_CODE void addItem (unsigned int i, const T& item)
	{
		const unsigned char* data = (const unsigned char*)&item;
		memcpy(buffer + (i * itemSize), data, itemSize);
	}

};

#endif