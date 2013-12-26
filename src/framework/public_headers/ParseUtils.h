#ifndef PARSEUTILS_H
#define PARSEUTILS_H

#include <StringUtils.h>

#include <vector_math.h>

#include <stdlib.h>
#include <exception>
#include <regex>
#include <string>
#include <vector>

class ParseUtils
{
public:
	//////////////////////////////////////////////////////////////////////////
	static vml_vec4 parseVec4(const std::string& aString)
	{
		std::vector<std::string> values;
		StringUtils::tokenize(aString, ",", values);

		if (values.size() != 4)
		{
			throw std::exception("parseVec4: invalid number of arguments");
		}

		std::string value = values[0];
		StringUtils::replace(value, "(", "");
		StringUtils::trim(value);
		float x = (float)atof(value.c_str());
		value = values[1];
		StringUtils::trim(value);
		float y = (float)atof(value.c_str());
		value = values[2];
		StringUtils::trim(value);
		float z = (float)atof(value.c_str());
		value = values[3];
		StringUtils::replace(value, ")", "");
		StringUtils::trim(value);
		float w = (float)atof(value.c_str());
		return vml_vec4(x, y, z, w);
	}

	//////////////////////////////////////////////////////////////////////////
	static vml_vec2 parseVec2(const std::string& aString)
	{
		std::vector<std::string> values;
		StringUtils::tokenize(aString, ",", values);

		if (values.size() != 2)
		{
			throw std::exception("parseVec2: invalid number of arguments");
		}

		std::string value = values[0];
		StringUtils::replace(value, "(", "");
		StringUtils::trim(value);
		float x = (float)atof(value.c_str());
		value = values[1];
		StringUtils::replace(value, ")", "");
		StringUtils::trim(value);
		float y = (float)atof(value.c_str());
		return vml_vec2(x, y);
	}

	//////////////////////////////////////////////////////////////////////////
	static void parseCommaSeparatedValues(const std::string& aString, float* out, unsigned int& size)
	{
		std::vector<std::string> values;
		StringUtils::tokenize(aString, ",", values);
		size = values.size();

		for (unsigned int i = 0; i < size; i++)
		{
			std::string value = values[i];
			StringUtils::replace(value, ",", "");
			StringUtils::trim(value);
			out[i] = (float)atof(value.c_str());
		}
	}

private:
	ParseUtils() {}
	~ParseUtils() {}

};

#endif