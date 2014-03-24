#include <Configuration.h>
#include <ConfigurationFunctions.h>
#include <StringUtils.h>
#include <Timer.h>

#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <exception>

const std::string RUN_BATCH_PATTERN = "@echo off\n" \
	"\n" \
	"setlocal ENABLEDELAYEDEXPANSION\n" \
	"\n" \
	"chdir ..\\..\\build\\mak.vc10\\x32\\src \n" \
	"\n" \
	"set EXEC_BIN=Release\\roadnetworkgenerator.exe\n" \
	"set CONFIGS_DIR=..\\..\\..\\..\\presentation\\configs\\#RUN_SUFFIX#\n" \
	"set FRAMES_DIR=..\\..\\..\\..\\presentation\\frames\\#RUN_SUFFIX#\n" \
	"\n" \
	"mkdir %FRAMES_DIR% \n" \
	"\n" \
	"set /a c=0 \n" \
	"for /l %%i in (1,1,#NUM_CONFIGS#) do ( \n" \
	"	set /a c=!c!+1 \n" \
	"	echo ********************** \n" \
	"	echo CONFIG. !c!/#NUM_CONFIGS# \n" \
	"	echo ********************** \n" \
	"	start /B /HIGH /WAIT %EXEC_BIN% 1024 768 %CONFIGS_DIR%\\#BASE_CONFIG_FILE#_%%i.config false true %FRAMES_DIR% \n" \
	") \n" \
	"\n" \
	"set /a c=0 \n" \
	"\n" \
	"chdir %FRAMES_DIR% \n" \
	"\n" \
	"for /f %%f in ('dir /b /o:d *.png') do ( \n" \
	"	ren %%f !c!.png \n" \
	"	set /a c=!c!+1 \n" \
	") \n" \
	"endlocal \n";

#define printField(__name1, __name2) \
	out << __name1 << "=" << configuration.##__name2 << std::endl

#define printStringField(__name1, __name2) \
	if (strlen(configuration.##__name2) > 0) \
	{ \
		out << __name1 << "=" << configuration.##__name2 << std::endl; \
	}

#define printBoolField(__name1, __name2) \
	out << __name1 << "=" << ((configuration.##__name2) ? "true" : "false") << std::endl

#define printVec4Field(__name1, __name2) \
	{ \
		vml_vec4 __vec = configuration.get##__name2##(); \
		out << __name1 << "="  << "(" << __vec.x << ", " << __vec.y << ", " << __vec.z << ", " << __vec.w << ")" << std::endl; \
	}

#define printVec3Field(__name1, __name2) \
	{ \
		vml_vec3 __vec = configuration.get##__name2##(); \
		out << __name1 << "=" << "(" << __vec.x << ", " << __vec.y << ", " << __vec.z << ")" << std::endl; \
	}

#define printQuatField(__name1, __name2) \
	{ \
		vml_quat __quat = configuration.get##__name2##(); \
		out << __name1 << "=" << "(" << __quat.x << ", " << __quat.y << ", " << __quat.z << ", " << __quat.w << ")" << std::endl; \
	}
	

void saveToFile(const Configuration& configuration, const std::string& outputFile)
{
	std::ofstream out;
	out.open(outputFile.c_str(), std::ios::out);

	if (!out.good())
	{
		throw std::exception((std::string("couldn't write to config. file: ") + outputFile).c_str());
	}

	printStringField("name", name);
	printField("seed", seed);
	printField("world_width", worldWidth);
	printField("world_height", worldHeight);
	printField("num_expansion_kernel_blocks", numExpansionKernelBlocks);
	printField("num_expansion_kernel_threads", numExpansionKernelThreads);
	printField("max_vertices", maxVertices);
	printField("max_edges", maxEdges);
	printField("highway_length", highwayLength);
	printField("street_length", streetLength);
	printField("max_street_branch_depth", maxStreetBranchDepth);
	printField("max_highway_branch_depth", maxHighwayBranchDepth);
	printField("max_street_obstacle_deviation_angle", maxStreetObstacleDeviationAngle);
	printField("highway_branching_distance", highwayBranchingDistance);
	printField("max_highway_derivation", maxHighwayDerivation);
	printField("max_street_derivation", maxStreetDerivation);
	printField("max_highway_goal_deviation", maxHighwayGoalDeviation);
	printField("min_sampling_weight", minSamplingWeight);
	printField("goal_distance_threshold", goalDistanceThreshold);
	printField("max_highway_obstacle_deviation_angle", maxHighwayObstacleDeviationAngle);
	printField("min_highway_length", minHighwayLength);
	printField("min_street_length", minStreetLength);
	printField("sampling_arc", samplingArc);
	printField("min_sampling_ray_length", minSamplingRayLength);
	printField("max_sampling_ray_length", maxSamplingRayLength);
	printField("quadtree_depth", quadtreeDepth);
	printField("snap_radius", snapRadius);
	printStringField("population_density_map", populationDensityMapFilePath);
	printStringField("water_bodies_map", waterBodiesMapFilePath);
	printStringField("blockades_map", blockadesMapFilePath);
	printStringField("natural_pattern_map", naturalPatternMapFilePath);
	printStringField("radial_pattern_map", radialPatternMapFilePath);
	printStringField("raster_pattern_map", rasterPatternMapFilePath);
	printBoolField("draw_spawn_point_labels", drawSpawnPointLabels);
	printBoolField("draw_graph_labels", drawGraphLabels);
	printBoolField("draw_quadtree", drawQuadtree);
	printBoolField("label_font_size", labelFontSize);
	printField("point_size", pointSize);
	printField("max_primitives", maxPrimitives);
	printField("min_block_area", minBlockArea);
	printField("vertex_buffer_size", vertexBufferSize);
	printField("index_buffer_size", indexBufferSize);
	std::stringstream sstream;
	sstream << "[";
	for (unsigned int i = 0, j = 0; i < configuration.numSpawnPoints; i++, j += 2)
	{
		sstream << "(" << configuration.spawnPointsData[j] << ", " << configuration.spawnPointsData[j + 1] << "), ";
	}
	std::string spawnPointsStr = sstream.str();
	spawnPointsStr = spawnPointsStr.substr(0, spawnPointsStr.size() - 2);
	spawnPointsStr += "]";
	out << "spawn_points=" << spawnPointsStr << std::endl;
	printBoolField("defines_camera_stats", definesCameraStats);
	printVec4Field("cycle_color", CycleColor);
	printVec4Field("filament_color", FilamentColor);
	printVec4Field("isolated_vertex_color", IsolatedVertexColor);
	printVec4Field("street_color", StreetColor);
	printVec4Field("quadtree_color", QuadtreeColor);
	printVec3Field("camera_position", CameraPosition);
	printQuatField("camera_rotation", CameraRotation);

	out.close();
}

std::string getConfigurationFileName(const std::string& outputFolder, const std::string& configurationName)
{
	return outputFolder + "/" + configurationName + ".config";
}

int main(int argc, char** argv)
{
	if (argc < 6)
	{
		std::cout << "command line options: <template configuration file> <final highway derivation> <final street derivation> <interval> <configs folder> <scripts folder>" << std::endl;
		exit(-1);
	}

	std::string templateConfigurationFile = argv[1];
	unsigned int finalHighwayDerivation = atoi(argv[2]);
	unsigned int finalStreetDerivation = atoi(argv[3]);
	unsigned int interval = atoi(argv[4]);
	std::string configsFolder = argv[5];
	std::string scriptsFolder = argv[6];

	try
	{
		std::string runSuffix = Timer::getTimestamp("_");

		configsFolder += "/" + runSuffix;

		if (!CreateDirectory(configsFolder.c_str(), NULL))
		{
			throw std::exception(std::string("couldn't create directory: " + configsFolder).c_str());
		}

		Configuration templateConfiguration;
		loadFromFile(templateConfiguration, templateConfigurationFile);

		std::string configurationName = templateConfiguration.name;

		unsigned int numConfigs = 0;
		for (unsigned int h = 0; h <= finalHighwayDerivation; h += interval)
		{
			Configuration newConfiguration;
			newConfiguration = templateConfiguration;
			std::stringstream sstream;
			sstream << configurationName << "_" << (++numConfigs);
			strcpy(newConfiguration.name, sstream.str().c_str());
			newConfiguration.maxHighwayDerivation = h;
			newConfiguration.maxStreetDerivation = 0;
			saveToFile(newConfiguration, getConfigurationFileName(configsFolder, newConfiguration.name));
		}

		for (unsigned int s = 0; s <= finalStreetDerivation; s += interval)
		{
			Configuration newConfiguration;
			newConfiguration = templateConfiguration;
			std::stringstream sstream;
			sstream << configurationName << "_" << (++numConfigs);
			strcpy(newConfiguration.name, sstream.str().c_str());
			newConfiguration.maxHighwayDerivation = finalHighwayDerivation;
			newConfiguration.maxStreetDerivation = s;
			saveToFile(newConfiguration, getConfigurationFileName(configsFolder, newConfiguration.name));
		}

		std::string scriptFile = scriptsFolder + "/RUN_" + runSuffix + ".BAT";

		std::ofstream out;
		out.open(scriptFile.c_str(), std::ios::out);

		if (!out.good())
		{
			throw std::exception((std::string("couldn't write to script file: ") + scriptFile).c_str());
		}

		std::string content = RUN_BATCH_PATTERN;
		StringUtils::replace(content, "#RUN_SUFFIX#", runSuffix);
		StringUtils::replace(content, "#BASE_CONFIG_FILE#", configurationName);
		std::stringstream sstream;
		sstream << numConfigs;
		StringUtils::replace(content, "#NUM_CONFIGS#", sstream.str());

		out << content;

		out.close();

		std::cout << numConfigs << " config(s) successfully created..." << std::endl;
	} 
	catch (std::exception& e)
	{
		std::cerr << "error: " << e.what() << std::endl;
	}

	system("pause");

	return -1;
}