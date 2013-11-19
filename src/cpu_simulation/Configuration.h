#ifndef CONFIGURATION_H
#define CONFIGURATION_H

#include <ImageMap.h>
#include <FileReader.h>
#include <StringUtils.h>

#include <glm/glm.hpp>

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
	int quadtreeCellArea;
	int quadtreeQueryRadius;
	ImageMap populationDensityMap;
	ImageMap waterBodiesMap;
	glm::vec4 highwayColor;
	glm::vec4 streetColor;
	glm::vec4 snapColor;
	glm::vec4 quadtreeColor;

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

			if (line.length() < 2 || (line[0] == '/' && line[1] == '/'))
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
		quadtreeCellArea = getPropertyAsInt(properties, "quadtree_cell_area");
		quadtreeQueryRadius = getPropertyAsInt(properties, "quadtree_query_radius");
		highwayColor = getPropertyAsColor(properties, "highway_color");
		streetColor = getPropertyAsColor(properties, "street_color");
		snapColor = getPropertyAsColor(properties, "snap_color");
		quadtreeColor = getPropertyAsColor(properties, "quadtree_color");

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

	glm::vec4 getPropertyAsColor(const std::map<std::string, std::string>& properties, const std::string& propertyName) 
	{
		std::string vectorStr = getProperty(properties, propertyName);
		std::vector<std::string> vectorComponentsStrs;
		StringUtils::tokenize(vectorStr, ",", vectorComponentsStrs);
		if (vectorComponentsStrs.size() != 4)
		{
			std::string errorMessage = "config: invalid color property ('" + propertyName + "')";
			throw std::exception(errorMessage.c_str());
		}

		std::string vectorComponentStr = vectorComponentsStrs[0];
		StringUtils::replace(vectorComponentStr, "(", "");
		StringUtils::trim(vectorComponentStr);
		float x = (float)atof(vectorComponentStr.c_str());

		vectorComponentStr = vectorComponentsStrs[1];
		StringUtils::trim(vectorComponentStr);
		float y = (float)atof(vectorComponentStr.c_str());

		vectorComponentStr = vectorComponentsStrs[2];
		StringUtils::trim(vectorComponentStr);
		float z = (float)atof(vectorComponentStr.c_str());

		vectorComponentStr = vectorComponentsStrs[3];
		StringUtils::replace(vectorComponentStr, ")", "");
		StringUtils::trim(vectorComponentStr);
		float w = (float)atof(vectorComponentStr.c_str());

		return glm::vec4(x, y, z, w);
	}


};

#endif