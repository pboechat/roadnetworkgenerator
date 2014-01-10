#ifndef RINGBUFFER_H
#define RINGBUFFER_H

template<typename T>
class RingBuffer
{
public:
	RingBuffer(T* data, unsigned int size) : data(data), _size(size), head(0), tail(0) {}
	~RingBuffer() {}

	/*inline T& operator[] (unsigned int i)
	{
		return data[i];
	}

	inline const T& operator[] (unsigned int i) const
	{
		return data[i];
	}*/

	T& peek()
	{
		return data[head];
	}

	bool pop(T& item)
	{
		if (head >= tail)
		{
			return false;
		}

		item = data[head++];
		return true;
	}

	bool pop()
	{
		if (head >= tail)
		{
			return false;
		}

		head++;
		return true;
	}

	bool push(T& item)
	{
		if (tail >= size)
		{
			return false;
		}

		data[tail++] = item;
		return true;
	}

	inline bool empty() const
	{
		return tail == head;
	}

	inline unsigned int size() const
	{
		return _size;
	}

private:
	T* data;
	unsigned int _size;
	unsigned int head;
	unsigned int tail;

};

#endif