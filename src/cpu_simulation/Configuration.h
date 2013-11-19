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
	int highwayWidth;
	int streetWidth;
	int maxDerivations;
	int deviationStep; // degrees
	int maxDeviation; // degrees
	int samplingArc; // degrees
	int quadtreeCellSize;
	int quadtreeQueryRadius;
	ImageMap populationDensityMap;
	ImageMap waterBodiesMap;

	Configuration() {}
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
		std::map<std::string, std::string> properties;

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
			properties.insert(std::make_pair(key, value));
		}

		worldWidth = getPropertyAsInt(properties, "world_width");
		worldHeight = getPropertyAsInt(properties, "world_height");
		highwayLength = getPropertyAsInt(properties, "highway_length");
		streetLength = getPropertyAsInt(properties, "street_length");
		highwayWidth = getPropertyAsInt(properties, "highway_width");
		streetWidth = getPropertyAsInt(properties, "street_width");
		maxDerivations = getPropertyAsInt(properties, "max_derivations");
		deviationStep = getPropertyAsInt(properties, "deviation_step");
		maxDeviation = getPropertyAsInt(properties, "max_deviation");
		samplingArc = getPropertyAsInt(properties, "sampling_arc");
		quadtreeCellSize = getPropertyAsInt(properties, "quadtree_cell_size");
		quadtreeQueryRadius = getPropertyAsInt(properties, "quadtree_query_radius");

		std::string populationDensityMapFile = getProperty(properties, "population_density_map");
		std::string waterBodiesMapFile = getProperty(properties, "water_bodies_map");

		populationDensityMap.import(populationDensityMapFile, worldWidth, worldHeight);
		waterBodiesMap.import(waterBodiesMapFile, worldWidth, worldHeight);
	}

private:
	static const std::string& getProperty(const std::map<std::string, std::string>& properties, const std::string& propertyName)
	{
		std::map<std::string, std::string>::const_iterator i;

		if ((i = properties.find(propertyName)) == properties.end())
		{
			std::string errorMessage = "config: '" + propertyName + "' not found";
			throw std::exception(errorMessage.c_str());
		}

		return i->second;
	}

	static int getPropertyAsInt(const std::map<std::string, std::string>& properties, const std::string& propertyName)
	{
		return atoi(getProperty(properties, propertyName).c_str());
	}

};

#endif