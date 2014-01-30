#ifndef ARRAY_H
#define ARRAY_H

#pragma once

#include <exception>

template<typename T>
class Array
{
public:
	Array() : data(0), capacity(0), count(0) {}
	Array(T* data, unsigned int capacity) : data(data), capacity(capacity), count(0) {}
	~Array() {}

	inline void setData(T* data, unsigned int capacity)
	{
		this->data = data;
		this->capacity = capacity;
	}

	inline void push(const T& item)
	{
		if (count >= capacity)
		{
			// FIXME: checking boundaries
			throw std::exception("Array: count >= capacity");
		}

		data[count++] = item;
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

	int indexOf(const T& item)
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

};

#endif