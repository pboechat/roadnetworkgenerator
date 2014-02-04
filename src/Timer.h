#ifndef TIMER_H
#define TIMER_H

#pragma once

#include <Windows.h>

class Timer
{
public:
	Timer() : _start(0), _end(0), _miliseconds(0)
	{
		LARGE_INTEGER frequency;

		if (QueryPerformanceFrequency(&frequency))
		{
			_miliseconds = 1000.0 / frequency.QuadPart;
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
			return (double)time.QuadPart * _miliseconds;
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
	double _miliseconds;

};

#endif
