#ifndef TIMER_H
#define TIMER_H

#include <Windows.h>

class Timer
{
public:
	Timer() : _start(0), _end(0), _seconds(0)
	{
		LARGE_INTEGER frequency;

		if (QueryPerformanceFrequency(&frequency))
		{
			_seconds = 1.0 / frequency.QuadPart;
		}

		else
		{
			throw std::exception("cannot query performance counter frequency");
		}
	}

	~Timer()
	{
	}

	inline double getTime()
	{
		LARGE_INTEGER time;

		if (QueryPerformanceCounter(&time))
		{
			return (double)time.QuadPart * _seconds;
		}

		else
		{
			throw std::exception("cannot query performance counter");
		}
	}

	inline void start()
	{
		_start = getTime();
	}

	inline void end()
	{
		_end = getTime();
	}

	inline double elapsedTime() const
	{
		return (_end - _start);
	}

private:
	double _start;
	double _end;
	double _seconds;

};

#endif
