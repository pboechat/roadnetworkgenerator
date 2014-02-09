#ifndef CONFIGURATIONFUNCTIONS_H
#define CONFIGURATIONFUNCTIONS_H

#include <Configuration.h>

#include <FileReader.h>
#include <StringUtils.h>
#include <ParseUtils.h>

#include <exception>
#include <time.h>
#include <random>
#include <regex>
#include <string>
#include <vector>
#include <map>

bool hasProperty(const std::map<std::string, std::string>& properties, const std::string& propertyName)
{
	return properties.find(propertyName) != properties.end();
}

const std::string& getProperty(const std::map<std::string, std::string>& properties, const std::string& propertyName)
{
	std::map<std::string, std::string>::const_iterator i;

	if ((i = properties.find(propertyName)) == properties.end())
	{
		std::string errorMessage = "config: '" + propertyName + "' not found";
		throw std::exception(errorMessage.c_str());
	}

	return i->second;
}

unsigned char getPropertyAsUnsignedChar(const std::map<std::string, std::string>& properties, const std::string& propertyName)
{
	return (unsigned char)atoi(getProperty(properties, propertyName).c_str());
}

unsigned int getPropertyAsUnsignedInt(const std::map<std::string, std::string>& properties, const std::string& propertyName)
{
	return (unsigned int)atoi(getProperty(properties, propertyName).c_str());
}

long getPropertyAsInt(const std::map<std::string, std::string>& properties, const std::string& propertyName)
{
	return atoi(getProperty(properties, propertyName).c_str());
}

float getPropertyAsFloat(const std::map<std::string, std::string>& properties, const std::string& propertyName)
{
	return (float)atof(getProperty(properties, propertyName).c_str());
}

bool getPropertyAsBool(const std::map<std::string, std::string>& properties, const std::string& propertyName)
{
	return getProperty(properties, propertyName) == "true";
}

vml_vec4 getPropertyAsVec4(const std::map<std::string, std::string>& properties, const std::string& propertyName)
{
	return ParseUtils::parseVec4(getProperty(properties, propertyName));
}

void getPropertyAsVec2Array(const std::map<std::string, std::string>& properties, const std::string& propertyName, vml_vec2* vec2Array, unsigned int& size, unsigned int maxSize)
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

void copyProperty(const std::map<std::string, std::string>& properties, const std::string& propertyName, char* dstBuffer, unsigned int bufferSize)
{
	std::string value;

	if (hasProperty(properties, propertyName))
	{
		value = getProperty(properties, propertyName);

		if (value.size() >= bufferSize)
		{
			throw std::exception("copyProperty: property size is greater than buffer size");
		}

		strncpy(dstBuffer, value.c_str(), value.size());
		dstBuffer[value.size()] = '\0';
	}

	else
	{
		dstBuffer[0] = '\0';
	}
}

void loadFromFile(Configuration& configuration, const std::string& filePath)
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

	copyProperty(properties, "name", configuration.name, MAX_CONFIGURATION_STRING_SIZE);
	configuration.seed = getPropertyAsInt(properties, "seed");

	if (configuration.seed < 0)
	{
		configuration.seed = (unsigned int)time(0);
	}

	srand(configuration.seed);
#ifdef USE_CUDA
	configuration.randZ = rand();
	configuration.randW = rand();
#endif
	configuration.worldWidth = getPropertyAsUnsignedInt(properties, "world_width");
	configuration.worldHeight = getPropertyAsUnsignedInt(properties, "world_height");
	configuration.maxVertices = getPropertyAsUnsignedInt(properties, "max_vertices");
	configuration.maxEdges = getPropertyAsUnsignedInt(properties, "max_edges");
	configuration.maxQuadrants = getPropertyAsUnsignedInt(properties, "max_quadrants");
	configuration.highwayLength = getPropertyAsUnsignedInt(properties, "highway_length");
	configuration.minSamplingRayLength = getPropertyAsUnsignedInt(properties, "max_sampling_ray_length");
	configuration.maxSamplingRayLength = getPropertyAsUnsignedInt(properties, "max_sampling_ray_length");
	configuration.streetLength = getPropertyAsUnsignedInt(properties, "street_length");
	configuration.maxStreetBranchDepth = getPropertyAsUnsignedInt(properties, "max_street_branch_depth");
	configuration.maxHighwayBranchDepth = getPropertyAsUnsignedInt(properties, "max_highway_branch_depth");
	configuration.minHighwayBranchingDistance = getPropertyAsUnsignedInt(properties, "min_highway_branching_distance");
	configuration.streetBranchingDelay = getPropertyAsUnsignedInt(properties, "street_branching_delay");
	configuration.maxHighwayDerivation = getPropertyAsUnsignedInt(properties, "max_highway_derivation");
	configuration.maxStreetDerivation = getPropertyAsUnsignedInt(properties, "max_street_derivation");
	configuration.maxHighwayGoalDeviation = getPropertyAsUnsignedInt(properties, "max_highway_goal_deviation");
	configuration.minSamplingWeight = getPropertyAsUnsignedInt(properties, "min_sampling_weight");
	configuration.goalDistanceThreshold = getPropertyAsUnsignedInt(properties, "goal_distance_threshold");
	configuration.halfMaxHighwayGoalDeviation = (configuration.maxHighwayGoalDeviation + 1) / 2;
	configuration.maxObstacleDeviationAngle = getPropertyAsUnsignedInt(properties, "max_obstacle_deviation_angle");
	configuration.minRoadLength = getPropertyAsUnsignedInt(properties, "min_road_length");
	configuration.samplingArc = getPropertyAsUnsignedInt(properties, "sampling_arc");
	configuration.halfSamplingArc = (configuration.samplingArc + 1) / 2;
	configuration.quadtreeDepth = getPropertyAsUnsignedInt(properties, "quadtree_depth");
	configuration.snapRadius = getPropertyAsFloat(properties, "snap_radius");
	configuration.cycleColor = getPropertyAsVec4(properties, "cycle_color");
	configuration.filamentColor = getPropertyAsVec4(properties, "filament_color");
	configuration.isolatedVertexColor = getPropertyAsVec4(properties, "isolated_vertex_color");
	configuration.streetColor = getPropertyAsVec4(properties, "street_color");
	configuration.quadtreeColor = getPropertyAsVec4(properties, "quadtree_color");
	configuration.drawLabels = getPropertyAsBool(properties, "draw_labels");
	configuration.maxPrimitives = getPropertyAsUnsignedInt(properties, "max_primitives");
	configuration.minBlockArea = getPropertyAsFloat(properties, "min_block_area");
	configuration.vertexBufferSize = getPropertyAsUnsignedInt(properties, "vertex_buffer_size");
	configuration.indexBufferSize = getPropertyAsUnsignedInt(properties, "index_buffer_size");
	getPropertyAsVec2Array(properties, "spawn_points", configuration.spawnPoints, configuration.numSpawnPoints, MAX_SPAWN_POINTS);
	copyProperty(properties, "population_density_map", configuration.populationDensityMapFilePath, MAX_CONFIGURATION_STRING_SIZE);
	copyProperty(properties, "water_bodies_map", configuration.waterBodiesMapFilePath, MAX_CONFIGURATION_STRING_SIZE);
	copyProperty(properties, "blockades_map", configuration.blockadesMapFilePath, MAX_CONFIGURATION_STRING_SIZE);
	copyProperty(properties, "natural_pattern_map", configuration.naturalPatternMapFilePath, MAX_CONFIGURATION_STRING_SIZE);
	copyProperty(properties, "radial_pattern_map", configuration.radialPatternMapFilePath, MAX_CONFIGURATION_STRING_SIZE);
	copyProperty(properties, "raster_pattern_map", configuration.rasterPatternMapFilePath, MAX_CONFIGURATION_STRING_SIZE);
}


#endif