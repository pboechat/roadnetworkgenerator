// memory leak detection
//#include <vld.h>

#include <RoadNetworkInputController.h>
#include <SceneRenderer.h>
#include <RoadNetworkGraphGenerator.h>
#include <RoadNetworkGeometryGenerator.h>
#include <RoadNetworkLabelsGenerator.h>
#include <Configuration.h>
#include <ConfigurationFunctions.h>
#include <ImageMap.h>
#include <Application.h>
#include <Camera.h>
#include <Box2D.h>
#include <VectorMath.h>
#include <ImageUtils.h>

#ifdef USE_CUDA
#include <cuda_runtime_api.h>
#endif
#include <string>
#include <iostream>
#include <io.h>
#include <iomanip>

//////////////////////////////////////////////////////////////////////////
#define loadImageMap(__map) \
	{ \
		if (strlen(configuration.##__map##FilePath) > 0) \
		{ \
			__map##Data = (unsigned char*)malloc(sizeof(unsigned char) * configuration.worldWidth * configuration.worldHeight); \
			ImageUtils::loadImage(configuration.##__map##FilePath, configuration.worldWidth, configuration.worldHeight, __map##Data); \
			__map.width = configuration.worldWidth; \
			__map.height = configuration.worldHeight; \
			__map.data = __map##Data; \
		} \
	}

//////////////////////////////////////////////////////////////////////////
#define SAFE_FREE_ON_HOST(__variable) \
	if (__variable != 0) \
	{ \
		free(__variable); \
	}

//////////////////////////////////////////////////////////////////////////
void printUsage()
{
	std::cerr << "Command line options: <width> <height> <configuration file>";
}

//////////////////////////////////////////////////////////////////////////
void generateAndDisplay(const std::string& configurationFile, SceneRenderer& renderer, RoadNetworkGeometryGenerator& geometryGenerator, RoadNetworkLabelsGenerator& labelsGenerator, Camera& camera)
{
	static bool firstTime = true;

	Configuration configuration;
	loadFromFile(configuration, configurationFile);

	std::cout << "seed: " << configuration.seed << std::endl;

	renderer.readConfigurations(configuration);

	geometryGenerator.readConfigurations(configuration);
	labelsGenerator.readConfigurations(configuration);

	unsigned char* populationDensityMapData = 0, *waterBodiesMapData = 0, *blockadesMapData = 0, *naturalPatternMapData = 0, *radialPatternMapData = 0, *rasterPatternMapData = 0;
	ImageMap populationDensityMap, waterBodiesMap, blockadesMap, naturalPatternMap, radialPatternMap, rasterPatternMap;
	loadImageMap(populationDensityMap);
	loadImageMap(waterBodiesMap);
	loadImageMap(blockadesMap);
	loadImageMap(naturalPatternMap);
	loadImageMap(radialPatternMap);
	loadImageMap(rasterPatternMap);

	RoadNetworkGraphGenerator graphGenerator(configuration, populationDensityMap, waterBodiesMap, blockadesMap, naturalPatternMap, radialPatternMap, rasterPatternMap);
	graphGenerator.addObserver(&geometryGenerator);
	graphGenerator.addObserver(&labelsGenerator);
	graphGenerator.execute();

	Box2D worldBounds(0.0f, 0.0f, (float)configuration.worldWidth, (float)configuration.worldHeight);
	renderer.setUpImageMaps(worldBounds, populationDensityMap, waterBodiesMap, blockadesMap);

	if (firstTime)
	{
		camera.centerOnTarget(worldBounds);
		firstTime = false;
	}

	SAFE_FREE_ON_HOST(populationDensityMapData);
	SAFE_FREE_ON_HOST(waterBodiesMapData);
	SAFE_FREE_ON_HOST(blockadesMapData);
	SAFE_FREE_ON_HOST(naturalPatternMapData);
	SAFE_FREE_ON_HOST(radialPatternMapData);
	SAFE_FREE_ON_HOST(rasterPatternMapData);
}

//////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
	int returnValue = -1;

#ifdef USE_CUDA
	bool cudaInitialized = false;
#endif

	try
	{
		if (argc < 4)
		{
			printUsage();
			exit(EXIT_FAILURE);
		}

		unsigned int screenWidth = (unsigned int)atoi(argv[1]);
		unsigned int screenHeight = (unsigned int)atoi(argv[2]);
		std::string configurationFile = argv[3];

		if (configurationFile.empty())
		{
			printUsage();
			exit(EXIT_FAILURE);
		}

#ifdef USE_CUDA
		Application application("Road Network Generator (GPU)", screenWidth, screenHeight);
#else
		Application application("Road Network Generator (CPU)", screenWidth, screenHeight);
#endif

		if (gl3wInit())
		{
			throw std::runtime_error("gl3wInit() failed");
		}

#ifdef USE_CUDA
		cudaDeviceProp deviceProperties;
		cudaGetDeviceProperties(&deviceProperties, 0);
		cudaSetDevice(0);
		cudaInitialized = true;
#endif

		Camera camera(screenWidth, screenHeight, FOVY_DEG, ZNEAR, ZFAR);
		RoadNetworkGeometryGenerator geometry;
		RoadNetworkLabelsGenerator labels;
		SceneRenderer renderer(camera, geometry, labels);
		RoadNetworkInputController inputController(camera, configurationFile, renderer, geometry, labels, generateAndDisplay);
		application.setCamera(camera);
		application.setRenderer(renderer);
		application.setInputController(inputController);
		generateAndDisplay(configurationFile, renderer, geometry, labels, camera);
		returnValue = application.run();
	}

	catch (std::exception& e)
	{
		std::cout << std::endl << "Exception: " << std::endl  << std::endl << e.what() << std::endl << std::endl;
		// DEBUG:
		system("pause");
	}

	catch (...)
	{
		std::cout << std::endl << "Unknown error" << std::endl << std::endl;
		// DEBUG:
		system("pause");
	}

#ifdef USE_CUDA
	if (cudaInitialized)
	{
		cudaDeviceReset();
	}
#endif

	return returnValue;
}
