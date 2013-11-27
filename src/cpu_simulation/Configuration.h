#ifndef CONFIGURATION_H
#define CONFIGURATION_H

#include <ImageMap.h>
#include <FileReader.h>
#include <StringUtils.h>
#include <ParseUtils.h>

#include <glm/glm.hpp>

#include <string>
#include <vector>
#include <map>
#include <exception>
#include <time.h>
#include <random>
#include <regex>
#ifdef _DEBUG
#include <iostream>
#endif

#define MAX_SPAWN_POINTS 10
#define VEC3_VECTOR_PATTERN "(\\([^\\)]+\\)\\,?)"

struct Configuration
{
	std::string name;
	int seed;
	unsigned int worldWidth;
	unsigned int worldHeight;
	unsigned int maxVertices;
	unsigned int maxEdges;
	unsigned int maxResultsPerQuery;
	unsigned int highwayLength;
	unsigned int minSamplingRayLength;
	unsigned int maxSamplingRayLength;
	unsigned int streetLength;
	unsigned int maxStreetBranchDepth;
	unsigned int highwayBranchingDelay;
	unsigned int minHighwayBranchingDistance;
	unsigned int minPureHighwayBranchingDistance;
	unsigned int streetBranchingDelay;
	unsigned int maxDerivations;
	unsigned int maxHighwayGoalDeviation; // degrees
	int halfMaxHighwayGoalDeviation; // degrees
	unsigned int maxObstacleDeviationAngle; // degrees
	unsigned int minRoadLength;
	unsigned int samplingArc; // degrees
	int halfSamplingArc;
	unsigned int quadtreeDepth;
	float snapRadius;
	ImageMap populationDensityMap;
	ImageMap waterBodiesMap;
	glm::vec4 highwayColor;
	glm::vec4 streetColor;
	glm::vec4 quadtreeColor;
	bool removeDeadEndRoads;
	unsigned int numSpawnPoints;
	glm::vec3 spawnPoints[MAX_SPAWN_POINTS];

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

		name = getProperty(properties, "name");
		seed = getPropertyAsInt(properties, "seed");
		if (seed < 0)
		{
			seed = (unsigned int)time(0);
		}
#ifdef _DEBUG
		std::cout << "seed: " << seed << std::endl;
#endif
		srand(seed);

		worldWidth = getPropertyAsUnsignedInt(properties, "world_width");
		worldHeight = getPropertyAsUnsignedInt(properties, "world_height");
		maxVertices = getPropertyAsUnsignedInt(properties, "max_vertices");
		maxEdges = getPropertyAsUnsignedInt(properties, "max_edges");
		maxResultsPerQuery = getPropertyAsUnsignedInt(properties, "max_results_per_query");
		highwayLength = getPropertyAsUnsignedInt(properties, "highway_length");
		minSamplingRayLength = getPropertyAsUnsignedInt(properties, "max_sampling_ray_length");
		maxSamplingRayLength = getPropertyAsUnsignedInt(properties, "max_sampling_ray_length");
		streetLength = getPropertyAsUnsignedInt(properties, "street_length");
		maxStreetBranchDepth = getPropertyAsUnsignedInt(properties, "max_street_branch_depth");
		highwayBranchingDelay = getPropertyAsUnsignedInt(properties, "highway_branching_delay");
		minHighwayBranchingDistance = getPropertyAsUnsignedInt(properties, "min_highway_branching_distance");
		minPureHighwayBranchingDistance = getPropertyAsUnsignedInt(properties, "min_pure_highway_branching_distance");
		streetBranchingDelay = getPropertyAsUnsignedInt(properties, "street_branching_delay");
		maxDerivations = getPropertyAsUnsignedInt(properties, "max_derivations");
		maxHighwayGoalDeviation = getPropertyAsUnsignedInt(properties, "max_highway_goal_deviation");
		halfMaxHighwayGoalDeviation = (maxHighwayGoalDeviation + 1) / 2;
		maxObstacleDeviationAngle = getPropertyAsUnsignedInt(properties, "max_obstacle_deviation_angle");
		minRoadLength = getPropertyAsUnsignedInt(properties, "min_road_length");
		samplingArc = getPropertyAsUnsignedInt(properties, "sampling_arc");
		halfSamplingArc = (samplingArc + 1) / 2;
		quadtreeDepth = getPropertyAsUnsignedInt(properties, "quadtree_depth");
		snapRadius = getPropertyAsFloat(properties, "snap_radius");
		highwayColor = getPropertyAsVec4(properties, "highway_color");
		streetColor = getPropertyAsVec4(properties, "street_color");
		quadtreeColor = getPropertyAsVec4(properties, "quadtree_color");
		std::string populationDensityMapFile = getProperty(properties, "population_density_map");
		std::string waterBodiesMapFile = getProperty(properties, "water_bodies_map");
		populationDensityMap.import(populationDensityMapFile, worldWidth, worldHeight);
		glm::vec4 color1 = getPropertyAsVec4(properties, "population_density_map_color1");
		glm::vec4 color2 = getPropertyAsVec4(properties, "population_density_map_color2");
		populationDensityMap.setColor1(color1);
		populationDensityMap.setColor2(color2);
		waterBodiesMap.import(waterBodiesMapFile, worldWidth, worldHeight);
		color1 = getPropertyAsVec4(properties, "water_bodies_map_color1");
		color2 = getPropertyAsVec4(properties, "water_bodies_map_color2");
		waterBodiesMap.setColor1(color1);
		waterBodiesMap.setColor2(color2);
		removeDeadEndRoads = getPropertyAsBool(properties, "remove_dead_end_roads");
		getPropertyAsVec3Array(properties, "spawn_points", spawnPoints, numSpawnPoints, MAX_SPAWN_POINTS);
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

	static unsigned int getPropertyAsUnsignedInt(const std::map<std::string, std::string>& properties, const std::string& propertyName)
	{
		return (unsigned int)atoi(getProperty(properties, propertyName).c_str());
	}

	static long getPropertyAsInt(const std::map<std::string, std::string>& properties, const std::string& propertyName)
	{
		return atoi(getProperty(properties, propertyName).c_str());
	}

	static float getPropertyAsFloat(const std::map<std::string, std::string>& properties, const std::string& propertyName)
	{
		return (float)atof(getProperty(properties, propertyName).c_str());
	}

	static bool getPropertyAsBool(const std::map<std::string, std::string>& properties, const std::string& propertyName)
	{
		return getProperty(properties, propertyName) == "true";
	}

	glm::vec4 getPropertyAsVec4(const std::map<std::string, std::string>& properties, const std::string& propertyName)
	{
		return ParseUtils::parseVec4(getProperty(properties, propertyName));
	}

	void getPropertyAsVec3Array(const std::map<std::string, std::string>& properties, const std::string& propertyName, glm::vec3* vec3Array, unsigned int& size, unsigned int maxSize)
	{
		std::string propertyValue = getProperty(properties, propertyName);

		std::smatch matches;
		size = 0;
		while (std::regex_search(propertyValue, matches, std::regex(VEC3_VECTOR_PATTERN)))
		{
			// FIXME: checking invariants
			if (size >= maxSize)
			{
				throw std::exception("i >= maxArraySize");
			}

			std::string vec3Str = matches[0].str();

			int pos = vec3Str.find_last_of(')');

			// FIXME: checking invariants
			if (pos == std::string::npos)
			{
				throw std::exception("pos == string::npos");
			}
			vec3Str = vec3Str.substr(0, pos + 1);

			vec3Array[size++] = ParseUtils::parseVec3(vec3Str);

			propertyValue = matches.suffix().str();
		}
	}

};

#endif