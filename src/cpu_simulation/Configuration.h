#ifndef CONFIGURATION_H
#define CONFIGURATION_H

#include <ImageMap.h>

#include <FileReader.h>
#include <StringUtils.h>

#include <string>
#include <vector>
#include <map>
#include <exception>

class Configuration
{
public:
	int worldWidth;
	int worldHeight;
	int highwayLength;
	int streetLength;
	int maxDerivations;
	int deviationStep;
	int maxDeviation;
	int samplingArc;
	ImageMap populationDensityMap;
	ImageMap waterBodiesMap;

	Configuration() : worldWidth(-1), worldHeight(-1) {}
	~Configuration() {}

	void loadFromFile(const std::string& filePath)
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

		worldWidth = atoi(find(keyValuePairs, "world_width").c_str());
		worldHeight = atoi(find(keyValuePairs, "world_height").c_str());
		highwayLength = atoi(find(keyValuePairs, "highway_length").c_str());
		streetLength = atoi(find(keyValuePairs, "street_length").c_str());
		maxDerivations = atoi(find(keyValuePairs, "max_derivations").c_str());
		deviationStep = atoi(find(keyValuePairs, "deviation_step").c_str());
		maxDeviation = atoi(find(keyValuePairs, "max_deviation").c_str());
		samplingArc = atoi(find(keyValuePairs, "sampling_arc").c_str());

		std::string populationDensityMapFile = find(keyValuePairs, "population_density_map");
		std::string waterBodiesMapFile = find(keyValuePairs, "water_bodies_map");

		populationDensityMap.import(populationDensityMapFile, worldWidth, worldHeight);
		waterBodiesMap.import(waterBodiesMapFile, worldWidth, worldHeight);
	}

private:
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