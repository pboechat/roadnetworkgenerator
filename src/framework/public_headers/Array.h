#ifndef ARRAY_H
#define ARRAY_H

#include <exception>

template<typename T>
class Array
{
public:
	Array(T* data, unsigned int capacity) : data(data), capacity(capacity), count(0) {}
	~Array() {}

	inline void push(T& item)
	{
		if (count >= capacity)
		{
			// FIXME: checking boundaries
			throw std::exception("count >= capacity");
		}

		data[count++] = item;
	}

	void remove(T& item)
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

	int indexOf(T& item)
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