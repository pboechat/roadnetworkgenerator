#ifndef CONFIGURATION_CUH
#define CONFIGURATION_CUH

#include "Defines.h"
#include <ImageMap.cuh>

#include <vector_math.h>

#include <string>
#include <vector>
#include <map>

struct Configuration
{
	char name[MAX_CONFIGURATION_STRING_SIZE];
	int seed;
	unsigned int worldWidth;
	unsigned int worldHeight;
	unsigned int maxVertices;
	unsigned int maxEdges;
	unsigned int maxResultsPerQuery;
	unsigned int maxQuadrants;
	//unsigned int maxWorkQueueCapacity;
	unsigned int highwayLength;
	unsigned int minSamplingRayLength;
	unsigned int maxSamplingRayLength;
	unsigned int streetLength;
	unsigned int maxStreetBranchDepth;
	unsigned int minHighwayBranchingDistance;
	unsigned int streetBranchingDelay;
	unsigned int maxHighwayDerivation;
	unsigned int maxStreetDerivation;
	unsigned int maxHighwayGoalDeviation; // degrees
	unsigned int goalDistanceThreshold;
	int halfMaxHighwayGoalDeviation; // degrees
	unsigned int maxObstacleDeviationAngle; // degrees
	unsigned int minRoadLength;
	unsigned int samplingArc; // degrees
	int halfSamplingArc; // degrees
	unsigned int quadtreeDepth;
	float snapRadius;
	char populationDensityMapFilePath[MAX_CONFIGURATION_STRING_SIZE];
	char waterBodiesMapFilePath[MAX_CONFIGURATION_STRING_SIZE];
	char blockadesMapFilePath[MAX_CONFIGURATION_STRING_SIZE];
	char naturalPatternMapFilePath[MAX_CONFIGURATION_STRING_SIZE];
	char radialPatternMapFilePath[MAX_CONFIGURATION_STRING_SIZE];
	char rasterPatternMapFilePath[MAX_CONFIGURATION_STRING_SIZE];
	vml_vec4 cycleColor;
	vml_vec4 filamentColor;
	vml_vec4 isolatedVertexColor;
	vml_vec4 streetColor;
	vml_vec4 quadtreeColor;
	bool drawLabels;
	unsigned int numSpawnPoints;
	vml_vec2 spawnPoints[MAX_SPAWN_POINTS];
	unsigned int maxPrimitives;
	unsigned int maxEdgeSequences;
	unsigned int maxVisitedVertices;
	float minBlockArea;
	unsigned int vertexBufferSize;
	unsigned int indexBufferSize;

	HOST_AND_DEVICE_CODE Configuration() {}
	HOST_AND_DEVICE_CODE ~Configuration() {}

	HOST_CODE void loadFromFile(const std::string& filePath);

private:
	static HOST_CODE bool hasProperty(const std::map<std::string, std::string>& properties, const std::string& propertyName);
	static HOST_CODE const std::string& getProperty(const std::map<std::string, std::string>& properties, const std::string& propertyName);
	static HOST_CODE unsigned char getPropertyAsUnsignedChar(const std::map<std::string, std::string>& properties, const std::string& propertyName);
	static HOST_CODE unsigned int getPropertyAsUnsignedInt(const std::map<std::string, std::string>& properties, const std::string& propertyName);
	static HOST_CODE long getPropertyAsInt(const std::map<std::string, std::string>& properties, const std::string& propertyName);
	static HOST_CODE float getPropertyAsFloat(const std::map<std::string, std::string>& properties, const std::string& propertyName);
	static HOST_CODE bool getPropertyAsBool(const std::map<std::string, std::string>& properties, const std::string& propertyName);
	static HOST_CODE vml_vec4 getPropertyAsVec4(const std::map<std::string, std::string>& properties, const std::string& propertyName);
	static HOST_CODE void getPropertyAsVec2Array(const std::map<std::string, std::string>& properties, const std::string& propertyName, vml_vec2* vec2Array, unsigned int& size, unsigned int maxSize);
	static HOST_CODE void copyProperty(const std::map<std::string, std::string>& properties, const std::string& propertyName, char* dstBuffer, unsigned int bufferSize);

};

#endif