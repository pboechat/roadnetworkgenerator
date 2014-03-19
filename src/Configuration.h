#ifndef CONFIGURATION_H
#define CONFIGURATION_H

#pragma once

#include <Constants.h>
#include <CpuGpuCompatibility.h>
#include <VectorMath.h>

struct Configuration
{
	char name[MAX_CONFIGURATION_STRING_SIZE];
	int seed;
	unsigned int worldWidth;
	unsigned int worldHeight;
	unsigned int numExpansionKernelBlocks;
	unsigned int numExpansionKernelThreads;
	unsigned int numCollisionDetectionKernelThreads;
	unsigned int maxVertices;
	unsigned int maxEdges;
	unsigned int totalNumQuadrants;
	unsigned int numLeafQuadrants;
	unsigned int highwayLength;
	unsigned int minSamplingRayLength;
	unsigned int maxSamplingRayLength;
	unsigned int streetLength;
	unsigned int maxStreetBranchDepth;
	unsigned int maxHighwayBranchDepth;
	unsigned int highwayBranchingDistance;
	unsigned int maxHighwayDerivation;
	unsigned int maxStreetDerivation;
	unsigned int maxHighwayGoalDeviation; // degrees
	int halfMaxHighwayGoalDeviation; // degrees
	unsigned int minSamplingWeight;
	unsigned int goalDistanceThreshold;
	unsigned int maxHighwayObstacleDeviationAngle; // degrees
	int halfMaxHighwayObstacleDeviationAngle; // degrees
	unsigned int maxStreetObstacleDeviationAngle; // degrees
	int halfMaxStreetObstacleDeviationAngle; // degrees
	unsigned int minHighwayLength;
	unsigned int minStreetLength;
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
	bool drawSpawnPointLabels;
	bool drawGraphLabels;
	bool drawQuadtree;
	float labelFontSize;
	float pointSize;
	unsigned int maxPrimitives;
	float minBlockArea;
	unsigned int vertexBufferSize;
	unsigned int indexBufferSize;
	unsigned int numSpawnPoints;
	float spawnPointsData[MAX_SPAWN_POINTS * 2];

	vec4FieldDeclaration(CycleColor, HOST_AND_DEVICE_CODE);
	vec4FieldDeclaration(FilamentColor, HOST_AND_DEVICE_CODE);
	vec4FieldDeclaration(IsolatedVertexColor, HOST_AND_DEVICE_CODE);
	vec4FieldDeclaration(StreetColor, HOST_AND_DEVICE_CODE);
	vec4FieldDeclaration(QuadtreeColor, HOST_AND_DEVICE_CODE);

	HOST_AND_DEVICE_CODE vml_vec2 getSpawnPoint(unsigned int i) const
	{
		unsigned int j = i << 1;
		return vml_vec2(spawnPointsData[j], spawnPointsData[j + 1]);
	}

	HOST_AND_DEVICE_CODE Configuration() {}
	HOST_AND_DEVICE_CODE ~Configuration() {}

#ifndef PARALLEL
	Configuration& operator = (const Configuration& other)
	{
		strcpy(name, other.name);
		seed = other.seed;
		worldWidth									= other.worldWidth								;
		worldHeight									= other.worldHeight								;
		numExpansionKernelBlocks					= other.numExpansionKernelBlocks					;
		numExpansionKernelThreads					= other.numExpansionKernelThreads					;
		numCollisionDetectionKernelThreads	= other.numCollisionDetectionKernelThreads;
		maxVertices									= other.maxVertices								;
		maxEdges									= other.maxEdges									;
		totalNumQuadrants							= other.totalNumQuadrants							;
		numLeafQuadrants							= other.numLeafQuadrants							;
		highwayLength								= other.highwayLength								;
		minSamplingRayLength						= other.minSamplingRayLength						;
		maxSamplingRayLength						= other.maxSamplingRayLength						;
		streetLength								= other.streetLength								;
		maxStreetBranchDepth						= other.maxStreetBranchDepth						;
		maxHighwayBranchDepth						= other.maxHighwayBranchDepth						;
		highwayBranchingDistance					= other.highwayBranchingDistance					;
		maxHighwayDerivation						= other.maxHighwayDerivation						;
		maxStreetDerivation							= other.maxStreetDerivation						;
		maxHighwayGoalDeviation						= other.maxHighwayGoalDeviation					;
		halfMaxHighwayGoalDeviation					= other.halfMaxHighwayGoalDeviation				;
		minSamplingWeight							= other.minSamplingWeight							;
		goalDistanceThreshold						= other.goalDistanceThreshold						;
		maxHighwayObstacleDeviationAngle			= other.maxHighwayObstacleDeviationAngle			;
		halfMaxHighwayObstacleDeviationAngle		= other.halfMaxHighwayObstacleDeviationAngle		;
		maxStreetObstacleDeviationAngle				= other.maxStreetObstacleDeviationAngle			;
		halfMaxStreetObstacleDeviationAngle			= other.halfMaxStreetObstacleDeviationAngle		;
		minHighwayLength							= other.minHighwayLength							;
		minStreetLength								= other.minStreetLength							;
		samplingArc									= other.samplingArc								;
		halfSamplingArc								= other.halfSamplingArc							;
		quadtreeDepth								= other.quadtreeDepth								;
		snapRadius									= other.snapRadius								;
		strcpy(populationDensityMapFilePath, other.populationDensityMapFilePath);
		strcpy(waterBodiesMapFilePath, other.waterBodiesMapFilePath);;
		strcpy(blockadesMapFilePath, other.blockadesMapFilePath);
		strcpy(naturalPatternMapFilePath, other.naturalPatternMapFilePath);
		strcpy(radialPatternMapFilePath, other.radialPatternMapFilePath);
		strcpy(rasterPatternMapFilePath, other.rasterPatternMapFilePath);
		drawSpawnPointLabels	= drawSpawnPointLabels	 ;
		drawGraphLabels			= drawGraphLabels		 ;
		drawQuadtree			= drawQuadtree			 ;
		labelFontSize			= labelFontSize			 ;
		pointSize				= pointSize				 ;
		maxPrimitives			= maxPrimitives			 ;
		minBlockArea			= minBlockArea			 ;
		vertexBufferSize		= vertexBufferSize		 ;
		indexBufferSize			= indexBufferSize		 ;
		numSpawnPoints			= numSpawnPoints		 ;
		memcpy(spawnPointsData, other.spawnPointsData, sizeof(float) * MAX_SPAWN_POINTS * 2);
		return *this;
	}
#endif

};

#endif