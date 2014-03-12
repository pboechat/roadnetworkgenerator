#ifndef TIMER_H
#define TIMER_H

#pragma once

#include <Windows.h>
#include <time.h>
#include <string>

class Timer
{
public:
	Timer() : _start(0), _end(0), _countsPerMiliseconds(0)
	{
		LARGE_INTEGER frequency;

		if (QueryPerformanceFrequency(&frequency))
		{
			_countsPerMiliseconds = (frequency.QuadPart / 1000.0f);
		}

		else
		{
			throw std::exception("cannot query performance counter frequency");
		}
	}

	~Timer()
	{
	}

	static std::string getTimestamp(const std::string& separator = ":") 
	{
		time_t t = time(0);
		char buffer[9] = {0};
		strftime(buffer, 9, (std::string("%H") + separator + "%M" + separator + "%S").c_str(), localtime(&t));
		return std::string(buffer);
	}

	inline double getTime()
	{
		LARGE_INTEGER time;

		if (QueryPerformanceCounter(&time))
		{
			return (double)time.QuadPart / _countsPerMiliseconds;
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
	double _countsPerMiliseconds;

};

#endif
