#ifndef CONFIGURATION_H
#define CONFIGURATION_H

#include <ImageMap.h>
#include <FileReader.h>
#include <StringUtils.h>
#include <ParseUtils.h>

#include <vector_math.h>

#include <string>
#include <vector>
#include <map>
#include <exception>
#include <time.h>
#include <random>
#include <regex>

#define MAX_SPAWN_POINTS 100
#define VEC2_VECTOR_PATTERN "(\\([^\\)]+\\)\\,?)"

struct Configuration
{
	std::string name;
	int seed;
	unsigned int worldWidth;
	unsigned int worldHeight;
	unsigned int maxVertices;
	unsigned int maxEdges;
	unsigned int maxResultsPerQuery;
	unsigned int maxWorkQueueCapacity;
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
	unsigned int goalDistanceThreshold;
	int halfMaxHighwayGoalDeviation; // degrees
	unsigned int maxObstacleDeviationAngle; // degrees
	unsigned int minRoadLength;
	unsigned int samplingArc; // degrees
	int halfSamplingArc; // degrees
	unsigned int quadtreeDepth;
	float snapRadius;
	ImageMap* populationDensityMap;
	ImageMap* waterBodiesMap;
	ImageMap* blockadesMap;
	ImageMap* naturalPatternMap;
	ImageMap* radialPatternMap;
	ImageMap* rasterPatternMap;
	vml_vec4 highwayColor;
	vml_vec4 streetColor;
	vml_vec4 quadtreeColor;
	bool removeDeadEndRoads;
	unsigned int numSpawnPoints;
	vml_vec2 spawnPoints[MAX_SPAWN_POINTS];

	Configuration() : populationDensityMap(0), waterBodiesMap(0), blockadesMap(0), naturalPatternMap(0), radialPatternMap(0), rasterPatternMap(0) {}
	~Configuration()
	{
		if (populationDensityMap != 0)
		{
			delete populationDensityMap;
		}

		if (waterBodiesMap != 0)
		{
			delete waterBodiesMap;
		}

		if (blockadesMap != 0)
		{
			delete blockadesMap;
		}

		if (naturalPatternMap != 0)
		{
			delete naturalPatternMap;
		}

		if (radialPatternMap != 0)
		{
			delete radialPatternMap;
		}

		if (rasterPatternMap != 0)
		{
			delete rasterPatternMap;
		}
	}

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

		srand(seed);
		worldWidth = getPropertyAsUnsignedInt(properties, "world_width");
		worldHeight = getPropertyAsUnsignedInt(properties, "world_height");
		maxVertices = getPropertyAsUnsignedInt(properties, "max_vertices");
		maxEdges = getPropertyAsUnsignedInt(properties, "max_edges");
		maxResultsPerQuery = getPropertyAsUnsignedInt(properties, "max_results_per_query");
		maxWorkQueueCapacity = getPropertyAsUnsignedInt(properties, "max_work_queue_capacity");
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
		goalDistanceThreshold = getPropertyAsUnsignedInt(properties, "goal_distance_threshold");
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
		populationDensityMap = createAndImportImageMap(properties, "population_density_map", worldWidth, worldHeight);
		waterBodiesMap = createAndImportImageMap(properties, "water_bodies_map", worldWidth, worldHeight);
		blockadesMap = createAndImportImageMap(properties, "blockades_map", worldWidth, worldHeight);
		naturalPatternMap = createAndImportImageMap(properties, "natural_pattern_map", worldWidth, worldHeight);
		radialPatternMap = createAndImportImageMap(properties, "radial_pattern_map", worldWidth, worldHeight);
		rasterPatternMap = createAndImportImageMap(properties, "raster_pattern_map", worldWidth, worldHeight);
		removeDeadEndRoads = getPropertyAsBool(properties, "remove_dead_end_roads");
		getPropertyAsVec2Array(properties, "spawn_points", spawnPoints, numSpawnPoints, MAX_SPAWN_POINTS);
	}

private:
	static bool hasProperty(const std::map<std::string, std::string>& properties, const std::string& propertyName)
	{
		return properties.find(propertyName) != properties.end();
	}

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

	static unsigned char getPropertyAsUnsignedChar(const std::map<std::string, std::string>& properties, const std::string& propertyName)
	{
		return (unsigned char)atoi(getProperty(properties, propertyName).c_str());
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

	static vml_vec4 getPropertyAsVec4(const std::map<std::string, std::string>& properties, const std::string& propertyName)
	{
		return ParseUtils::parseVec4(getProperty(properties, propertyName));
	}

	static void getPropertyAsVec2Array(const std::map<std::string, std::string>& properties, const std::string& propertyName, vml_vec2* vec2Array, unsigned int& size, unsigned int maxSize)
	{
		std::string propertyValue = getProperty(properties, propertyName);
		std::smatch matches;
		size = 0;

		while (std::regex_search(propertyValue, matches, std::regex(VEC2_VECTOR_PATTERN)))
		{
			// FIXME: checking invariants
			if (size >= maxSize)
			{
				throw std::exception("i >= maxArraySize");
			}

			std::string vec2Str = matches[0].str();
			int pos = vec2Str.find_last_of(')');

			// FIXME: checking invariants
			if (pos == std::string::npos)
			{
				throw std::exception("pos == string::npos");
			}

			vec2Str = vec2Str.substr(0, pos + 1);
			vec2Array[size++] = ParseUtils::parseVec2(vec2Str);
			propertyValue = matches.suffix().str();
		}
	}

	static ImageMap* createAndImportImageMap(const std::map<std::string, std::string>& properties, const std::string& propertyName, unsigned int width, unsigned int height)
	{
		if (!hasProperty(properties, propertyName))
		{
			return 0;
		}

		std::string mapFile = getProperty(properties, propertyName);
		ImageMap* imageMap = new ImageMap();
		imageMap->import(mapFile, width, height);
		vml_vec4 color1 = getPropertyAsVec4(properties, propertyName + "_color1");
		vml_vec4 color2 = getPropertyAsVec4(properties, propertyName + "_color2");
		imageMap->setColor1(color1);
		imageMap->setColor2(color2);
		return imageMap;
	}

};

#endif