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
	unsigned int numExpansionKernelBlocks;
	unsigned int numExpansionKernelThreads;
	unsigned int numCollisionDetectionKernelBlocksPerQuadrant;
	unsigned int numCollisionDetectionKernelThreads;
	unsigned int maxVertices;
	unsigned int maxEdges;
	unsigned int maxQuadrants;
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
	vml_vec4 cycleColor;
	vml_vec4 filamentColor;
	vml_vec4 isolatedVertexColor;
	vml_vec4 streetColor;
	vml_vec4 quadtreeColor;
	bool drawSpawnPointLabels;
	bool drawGraphLabels;
	bool drawQuadtree;
	float labelFontSize;
	float pointSize;
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