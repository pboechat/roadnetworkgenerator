#ifndef ARRAY_H
#define ARRAY_H

#include "Defines.h"

template<typename T>
HOST_AND_DEVICE_CODE class Array
{
public:
	HOST_AND_DEVICE_CODE Array() : data(0), capacity(0), count(0) {}
	HOST_AND_DEVICE_CODE Array(T* data, unsigned int capacity) : data(data), capacity(capacity), count(0) {}
	HOST_AND_DEVICE_CODE ~Array() {}

	inline HOST_AND_DEVICE_CODE void setData(T* data, unsigned int capacity)
	{
		this->data = data;
		this->capacity = capacity;
	}

	inline HOST_AND_DEVICE_CODE void push(const T& item)
	{
		if (count >= capacity)
		{
			// FIXME: checking boundaries
			//throw std::exception("Array: count >= capacity");
			THROW_EXCEPTION("Array: count >= capacity");
		}

		data[count++] = item;
	}

	HOST_AND_DEVICE_CODE void remove(const T& item)
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

	HOST_AND_DEVICE_CODE int indexOf(const T& item)
	{
		for (unsigned int i = 0; i < count; i++)
		{
			if (data[i] == item)
			{
				return i;
			}
		}

		return -1;
	}

	inline HOST_AND_DEVICE_CODE unsigned int size() const
	{
		return count;
	}

	inline HOST_AND_DEVICE_CODE T& operator[] (unsigned int i)
	{
		return data[i];
	}

	inline HOST_AND_DEVICE_CODE const T& operator[] (unsigned int i) const
	{
		return data[i];
	}

private:
	T* data;
	unsigned int capacity;
	unsigned int count;

};

#endif