#ifndef CONFIG_H
#define CONFIG_H

#include <FileReader.h>
#include <StringUtils.h>

#include <string>
#include <vector>
#include <map>
#include <exception>

class Config
{
public:
	std::string populationMapFile;
	std::string waterBodiesMapFile;

	~Config() {}

	static Config* loadFromFile(const std::string& filePath)
	{
		std::string fileContent = FileReader::read(filePath);

		if (fileContent.empty())
		{
			throw std::exception("config: empty file");
		}

		std::vector<std::string> lines;
		StringUtils::tokenize(fileContent, "\n", lines);
		std::map<std::string, std::string> keyValuePairs;

		for (unsigned int i = 0; i < lines.size(); i++)
		{
			std::string line = lines[i];

			if (line.length() < 2 || (line[0] == '\\' && line[1] == '\\'))
			{
				continue;
			}

			std::vector<std::string> parts;
			StringUtils::tokenize(line, "=", parts);

			if (parts.size() != 2)
			{
				throw std::exception("config: malformed key/value pair");
			}

			std::string key = parts[0];
			StringUtils::trim(key);
			std::string value = parts[1];
			StringUtils::trim(value);
			keyValuePairs.insert(std::make_pair(key, value));
		}

		std::string populationMapFile = find(keyValuePairs, "population_map");
		std::string waterBodiesMapFile = find(keyValuePairs, "waterbodies_map");

		Config* config = new Config();
		config->populationMapFile = populationMapFile;
		config->waterBodiesMapFile = waterBodiesMapFile;
		return config;
	}

private:
	Config() {}

	static const std::string& find(const std::map<std::string, std::string>& keyValuePairs, const std::string& key)
	{
		std::map<std::string, std::string>::const_iterator i;

		if ((i = keyValuePairs.find(key)) == keyValuePairs.end())
		{
			std::string errorMessage = "config: '" + key + "' not found";
			throw std::exception(errorMessage.c_str());
		}

		return i->second;
	}

};

#endif