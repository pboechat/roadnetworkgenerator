#ifndef STRINGUTILS_H
#define STRINGUTILS_H

#include <vector>
#include <string>
#include <cctype>
#include <algorithm>

class StringUtils
{
public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	static void tokenize(const std::string& aString, const std::string& delimiter, std::vector<std::string>& tokens)
	{
		unsigned int currentPosition = 0;
		unsigned int nextPosition = std::string::npos;

		while (currentPosition != std::string::npos)
		{
			nextPosition = aString.find_first_of(delimiter, currentPosition);

			if (nextPosition != currentPosition)
			{
				std::string token = aString.substr(currentPosition, nextPosition - currentPosition);
				tokens.push_back(token);
			}

			currentPosition = aString.find_first_not_of(delimiter, nextPosition);
		}
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	static void trimLeft(std::string& aString)
	{
		aString.erase(aString.begin(), std::find_if(aString.begin(), aString.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	static void trimRight(std::string& aString)
	{
		aString.erase(std::find_if(aString.rbegin(), aString.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), aString.end());
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	static void trim(std::string& aString)
	{
		trimLeft(aString);
		trimRight(aString);
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	static bool notContains(const std::string& a, const std::string& b)
	{
		return a.find(b) == std::string::npos;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	static void replace(std::string& aString, const std::string& from, const std::string& to)
	{
		unsigned int startPosition;

		while ((startPosition = aString.find(from)) != std::string::npos)
		{
			aString.replace(startPosition, from.length(), to);
		}
	}

private:
	StringUtils() {}
	~StringUtils() {}

};

#endif