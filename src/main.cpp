// memory leak detection
//#include <vld.h>

#include <GlobalVariables.h>
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
#include <Log.h>

#ifdef PARALLEL
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

#define IS_TRUE(__x) strcmp(__x, "true") == 0

//////////////////////////////////////////////////////////////////////////
void printBasicUsage()
{
	Log::logger("default") << "basic command line options: <width> <height> <configuration file>";
}

void printAdvancedUsage()
{
	Log::logger("default") << "adv. command line options: <width> <height> <configuration file> <dump statistics> <dump first frame> <dump folder>";
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
		if (configuration.definesCameraStats)
		{
			camera.localTransform.position = configuration.getCameraPosition();
			camera.localTransform.rotation = configuration.getCameraRotation();
		}
		else
		{
			camera.centerOnTarget(worldBounds);
		}
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
#ifdef PARALLEL
	bool cudaInitialized = false;
#endif
	unsigned int screenWidth;
	unsigned int screenHeight;
	std::string configurationFile;

	Log::addLogger("default", new ConsoleLogger());

	if (argc < 4)
	{
		printBasicUsage();
		goto exit;
	}

	screenWidth = (unsigned int)atoi(argv[1]);
	screenHeight = (unsigned int)atoi(argv[2]);
	configurationFile = argv[3];

	if (configurationFile.empty())
	{
		printBasicUsage();
		goto exit;
	}

	if (argc > 4)
	{
		if (argc < 6)
		{
			printAdvancedUsage();
			goto exit;
		}

		g_dumpStatistics = IS_TRUE(argv[4]);
		g_dumpFirstFrame = IS_TRUE(argv[5]);
		g_dumpFolder = argv[6];

		if (g_dumpStatistics)
		{
			Log::addLogger("statistics", new CSVLogger(g_dumpFolder + "/statistics.csv"));
		}
		
		Log::addLogger("error", new MultiLogger(2, new ConsoleLogger(), new FileLogger("error.log")));
	}
	else
	{
		Log::addLogger("error", new ConsoleLogger());
	}

	try
	{
#ifdef PARALLEL
		Application application("Road Network Generator (GPU)", screenWidth, screenHeight);
#else
		Application application("Road Network Generator (CPU)", screenWidth, screenHeight);
#endif

		if (gl3wInit())
		{
			throw std::runtime_error("gl3wInit() failed");
		}

#ifdef PARALLEL
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

		if (!g_dumpStatistics || g_dumpFirstFrame)
		{
			returnValue = application.run(g_dumpFirstFrame);
		}
	}

	catch (std::exception& e)
	{
		Log::logger("error") << Logger::endl << "Exception: " << Logger::endl  << Logger::endl << e.what() << Logger::endl << Logger::endl;
		if (!g_dumpStatistics && !g_dumpFirstFrame)
		{
			system("pause");
		}
	}

	catch (...)
	{
		Log::logger("error") << Logger::endl << "Unknown error" << Logger::endl << Logger::endl;
		if (!g_dumpStatistics && !g_dumpFirstFrame)
		{
			system("pause");
		}
	}

exit:

#ifdef PARALLEL
	if (cudaInitialized)
	{
		cudaDeviceReset();
	}
#endif

	Log::dispose();

	return returnValue;
}
