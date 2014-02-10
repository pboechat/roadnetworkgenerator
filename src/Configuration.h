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
#ifdef USE_CUDA
	unsigned int randZ;
	unsigned int randW;
#endif
	unsigned int worldWidth;
	unsigned int worldHeight;
	unsigned int maxVertices;
	unsigned int maxEdges;
	unsigned int maxQuadrants;
	unsigned int highwayLength;
	unsigned int minSamplingRayLength;
	unsigned int maxSamplingRayLength;
	unsigned int streetLength;
	unsigned int maxStreetBranchDepth;
	unsigned int maxHighwayBranchDepth;
	unsigned int minHighwayBranchingDistance;
	unsigned int streetBranchingDelay;
	unsigned int maxHighwayDerivation;
	unsigned int maxStreetDerivation;
	unsigned int maxHighwayGoalDeviation; // degrees
	unsigned int minSamplingWeight;
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
	float minBlockArea;
	unsigned int vertexBufferSize;
	unsigned int indexBufferSize;

	HOST_AND_DEVICE_CODE Configuration() {}
	HOST_AND_DEVICE_CODE ~Configuration() {}

};

#endif