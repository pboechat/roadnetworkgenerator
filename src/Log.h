#ifndef LOG_H
#define LOG_H

#include <StringUtils.h>

#include <string>
#include <cstdarg>
#include <exception>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <map>

//////////////////////////////////////////////////////////////////////////
class Logger
{
public:
	virtual ~Logger()
	{
	}

	virtual bool firstUse() const = 0;

	virtual void print(const std::string& arg0) = 0;
	virtual void print(const char* arg0) = 0;
	virtual void print(bool arg0) = 0;
	virtual void print(unsigned char arg0) = 0;
	virtual void print(char arg0) = 0;
	virtual void print(unsigned short arg0) = 0;
	virtual void print(short arg0) = 0;
	virtual void print(unsigned int arg0) = 0;
	virtual void print(int arg0) = 0;
	virtual void print(unsigned long arg0) = 0;
	virtual void print(long arg0) = 0;
	virtual void print(float arg0) = 0;
	virtual void print(double arg0) = 0;

	static Logger& endl(Logger& arg0)
	{
		arg0.printEndLine();
		return arg0;
	}

	virtual void printEndLine() = 0;

};

//////////////////////////////////////////////////////////////////////////
static Logger& operator << (Logger& arg0, Logger& (*manipulator)(Logger&))
{
	return manipulator(arg0);
}

//////////////////////////////////////////////////////////////////////////
static Logger& operator << (Logger& arg0, const std::string& arg1)
{
	arg0.print(arg1);
	return arg0;
}

//////////////////////////////////////////////////////////////////////////
static Logger& operator << (Logger& arg0, const char* arg1)
{
	arg0.print(arg1);
	return arg0;
}

//////////////////////////////////////////////////////////////////////////
static Logger& operator << (Logger& arg0, bool arg1)
{
	arg0.print(arg1);
	return arg0;
}

//////////////////////////////////////////////////////////////////////////
static Logger& operator << (Logger& arg0, unsigned char arg1)
{
	arg0.print(arg1);
	return arg0;
}

//////////////////////////////////////////////////////////////////////////
static Logger& operator << (Logger& arg0, char arg1)
{
	arg0.print(arg1);
	return arg0;
}

//////////////////////////////////////////////////////////////////////////
static Logger& operator << (Logger& arg0, unsigned short arg1)
{
	arg0.print(arg1);
	return arg0;
}

//////////////////////////////////////////////////////////////////////////
static Logger& operator << (Logger& arg0, short arg1)
{
	arg0.print(arg1);
	return arg0;
}

//////////////////////////////////////////////////////////////////////////
static Logger& operator << (Logger& arg0, unsigned int arg1)
{
	arg0.print(arg1);
	return arg0;
}

//////////////////////////////////////////////////////////////////////////
static Logger& operator << (Logger& arg0, int arg1)
{
	arg0.print(arg1);
	return arg0;
}

//////////////////////////////////////////////////////////////////////////
static Logger& operator << (Logger& arg0, unsigned long arg1)
{
	arg0.print(arg1);
	return arg0;
}

//////////////////////////////////////////////////////////////////////////
static Logger& operator << (Logger& arg0, long arg1)
{
	arg0.print(arg1);
}

//////////////////////////////////////////////////////////////////////////
static Logger& operator << (Logger& arg0, float arg1)
{
	arg0.print(arg1);
	return arg0;
}

//////////////////////////////////////////////////////////////////////////
static Logger& operator << (Logger& arg0, double arg1)
{
	arg0.print(arg1);
	return arg0;
}

//////////////////////////////////////////////////////////////////////////
class NullLogger : public Logger
{
public:
	virtual bool firstUse() const
	{
		return true;
	}

	virtual void print(const std::string& arg0)
	{
	}

	virtual void print(const char* arg0)
	{
	}

	virtual void print(bool arg0)
	{
	}

	virtual void print(unsigned char arg0)
	{
	}

	virtual void print(char arg0)
	{
	}

	virtual void print(unsigned short arg0)
	{
	}

	virtual void print(short arg0)
	{
	}

	virtual void print(unsigned int arg0)
	{
	}

	virtual void print(int arg0)
	{
	}

	virtual void print(unsigned long arg0)
	{
	}

	virtual void print(long arg0)
	{
	}

	virtual void print(float arg0)
	{
	}

	virtual void print(double arg0)
	{
	}

	virtual void printEndLine()
	{
	}

};

//////////////////////////////////////////////////////////////////////////
class MultiLogger : public Logger
{
public:
	MultiLogger(unsigned int numLoggers, Logger* logger, ...)
	{
		va_list arguments;
		va_start(arguments, numLoggers);
		for (unsigned int i = 0; i < numLoggers; i++)
		{
			others.push_back(va_arg(arguments, Logger*));
		}
		va_end (arguments);
	}

	virtual ~MultiLogger()
	{
		for (unsigned int i = 0; i < others.size(); i++)
		{
			delete others[i];
		}
		others.clear();
	}

	virtual bool firstUse() const
	{
		for (unsigned int i = 0; i < others.size(); i++)
		{
			if (!others[i]->firstUse())
			{
				return false;
			}
		}
		return true;
	}

	virtual void print(const std::string& arg0)
	{
		for (unsigned int i = 0; i < others.size(); i++)
		{
			others[i]->print(arg0);
		}
	}

	virtual void print(const char* arg0)
	{
		for (unsigned int i = 0; i < others.size(); i++)
		{
			others[i]->print(arg0);
		}
	}

	virtual void print(bool arg0)
	{
		for (unsigned int i = 0; i < others.size(); i++)
		{
			others[i]->print(arg0);
		}
	}

	virtual void print(unsigned char arg0)
	{
		for (unsigned int i = 0; i < others.size(); i++)
		{
			others[i]->print(arg0);
		}
	}

	virtual void print(char arg0)
	{
		for (unsigned int i = 0; i < others.size(); i++)
		{
			others[i]->print(arg0);
		}
	}

	virtual void print(unsigned short arg0)
	{
		for (unsigned int i = 0; i < others.size(); i++)
		{
			others[i]->print(arg0);
		}
	}

	virtual void print(short arg0)
	{
		for (unsigned int i = 0; i < others.size(); i++)
		{
			others[i]->print(arg0);
		}
	}

	virtual void print(unsigned int arg0)
	{
		for (unsigned int i = 0; i < others.size(); i++)
		{
			others[i]->print(arg0);
		}
	}

	virtual void print(int arg0)
	{
		for (unsigned int i = 0; i < others.size(); i++)
		{
			others[i]->print(arg0);
		}
	}

	virtual void print(unsigned long arg0)
	{
		for (unsigned int i = 0; i < others.size(); i++)
		{
			others[i]->print(arg0);
		}
	}

	virtual void print(long arg0)
	{
		for (unsigned int i = 0; i < others.size(); i++)
		{
			others[i]->print(arg0);
		}
	}

	virtual void print(float arg0)
	{
		for (unsigned int i = 0; i < others.size(); i++)
		{
			others[i]->print(arg0);
		}
	}

	virtual void print(double arg0)
	{
		for (unsigned int i = 0; i < others.size(); i++)
		{
			others[i]->print(arg0);
		}
	}

	virtual void printEndLine()
	{
		for (unsigned int i = 0; i < others.size(); i++)
		{
			others[i]->printEndLine();
		}
	}

private:
	std::vector<Logger*> others;
	
};

//////////////////////////////////////////////////////////////////////////
class ConsoleLogger : public Logger
{
public:
	virtual bool firstUse() const
	{
		return true;
	}

	virtual void print(const std::string& arg0)
	{
		std::cout << arg0;
	}

	virtual void print(const char* arg0)
	{
		std::cout << arg0;
	}

	virtual void print(bool arg0)
	{
		std::cout << arg0;
	}

	virtual void print(unsigned char arg0)
	{
		std::cout << arg0;
	}

	virtual void print(char arg0)
	{
		std::cout << arg0;
	}

	virtual void print(unsigned short arg0)
	{
		std::cout << arg0;
	}

	virtual void print(short arg0)
	{
		std::cout << arg0;
	}

	virtual void print(unsigned int arg0)
	{
		std::cout << arg0;
	}

	virtual void print(int arg0)
	{
		std::cout << arg0;
	}

	virtual void print(unsigned long arg0)
	{
		std::cout << arg0;
	}

	virtual void print(long arg0)
	{
		std::cout << arg0;
	}

	virtual void print(float arg0)
	{
		std::cout << arg0;
	}

	virtual void print(double arg0)
	{
		std::cout << arg0;
	}

	virtual void printEndLine()
	{
		std::cout << std::endl;
	}

};

//////////////////////////////////////////////////////////////////////////
class FileLogger : public Logger
{
public:
	FileLogger(const std::string& fileName)
	{
		created = !fileExists(fileName);
		file.open(fileName.c_str(), std::ios::out | std::ios::app);
	}

	virtual ~FileLogger()
	{
		file.close();
	}

	virtual bool firstUse() const
	{
		return created;
	}

	virtual void print(const std::string& arg0)
	{
		file << arg0;
	}

	virtual void print(const char* arg0)
	{
		file << arg0;
	}

	virtual void print(bool arg0)
	{
		file << arg0;
	}

	virtual void print(unsigned char arg0)
	{
		file << arg0;
	}

	virtual void print(char arg0)
	{
		file << arg0;
	}

	virtual void print(unsigned short arg0)
	{
		file << arg0;
	}

	virtual void print(short arg0)
	{
		file << arg0;
	}

	virtual void print(unsigned int arg0)
	{
		file << arg0;
	}

	virtual void print(int arg0)
	{
		file << arg0;
	}

	virtual void print(unsigned long arg0)
	{
		file << arg0;
	}

	virtual void print(long arg0)
	{
		file << arg0;
	}

	virtual void print(float arg0)
	{
		file << arg0;
	}

	virtual void print(double arg0)
	{
		file << arg0;
	}

	virtual void printEndLine()
	{
		file << s_lineEnd;
	}

	static void setLineEnd(const std::string& lineEnd)
	{
		s_lineEnd = lineEnd;
	}

protected:
	std::ofstream file;

private:
	static std::string s_lineEnd;
	bool created;

	bool fileExists(const std::string& fileName)
	{
		FILE* file = fopen(fileName.c_str(), "rt");
		bool exists = (file != 0);
		if (exists)
		{
			fclose(file);
		}
		return exists;
	}

};

//////////////////////////////////////////////////////////////////////////
class CSVLogger : public FileLogger
{
public:
	CSVLogger(const std::string& fileName) : FileLogger(fileName)
	{
	}

	virtual void print(const std::string& arg0)
	{
		file << arg0 << ";";
	}

	virtual void print(const char* arg0)
	{
		file << arg0 << ";";
	}

	virtual void print(bool arg0)
	{
		file << arg0 << ";";
	}

	virtual void print(unsigned char arg0)
	{
		file << arg0 << ";";
	}

	virtual void print(char arg0)
	{
		file << arg0 << ";";
	}

	virtual void print(unsigned short arg0)
	{
		file << arg0 << ";";
	}

	virtual void print(short arg0)
	{
		file << arg0 << ";";
	}

	virtual void print(unsigned int arg0)
	{
		file << arg0 << ";";
	}

	virtual void print(int arg0)
	{
		file << arg0 << ";";
	}

	virtual void print(unsigned long arg0)
	{
		file << arg0 << ";";
	}

	virtual void print(long arg0)
	{
		file << arg0 << ";";
	}

	virtual void print(float arg0)
	{
		file << useComma(arg0) << ";";
	}

	virtual void print(double arg0)
	{
		file << useComma(arg0) << ";";
	}

private:
	template<typename T>
	std::string useComma(T arg0)
	{
		std::stringstream sstream;
		sstream << arg0;
		std::string str = sstream.str();
		StringUtils::replace(str, ".", ",");
		return str;
	}

};

//////////////////////////////////////////////////////////////////////////
class Log
{
public:
	static void addLogger(const std::string& name, Logger* logger)
	{
		s_loggers.insert(std::make_pair(name, logger));
	}

	static void print(const std::string& loggerName, const std::string& message)
	{
		Logger* logger = findLogger(loggerName);
		if (logger != 0)
		{
			(*logger) << message;
		}
	}

	static Logger& logger(const std::string& loggerName)
	{
		Logger* logger = findLogger(loggerName);
		if (logger == 0)
		{
			throw std::exception((std::string("Logger not found: ") + loggerName).c_str());
		}
		return *logger;
	}

	static void dispose()
	{
		std::map<std::string, Logger*>::iterator it = s_loggers.begin();
		while (it != s_loggers.end())
		{
			delete it->second;
			it++;
		}
		s_loggers.clear();
	}

private:
	static std::map<std::string, Logger*> s_loggers;

	static Logger* findLogger(const std::string& loggerName)
	{
		std::map<std::string, Logger*>::iterator it = s_loggers.find(loggerName);
		if (it != s_loggers.end())
		{
			return it->second;
		}
		return 0;
	}

};

#endif