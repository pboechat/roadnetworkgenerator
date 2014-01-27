#include "Defines.h"
#include <Globals.h>
#include <Road.h>
#include <Branch.h>

#include <FreeImage.h>

//////////////////////////////////////////////////////////////////////////
DEVICE_CODE Configuration* g_dConfiguration = 0;
//////////////////////////////////////////////////////////////////////////
HOST_CODE Configuration* g_hConfiguration = 0;
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE RoadNetworkGraph::Graph* g_dGraph = 0;
//////////////////////////////////////////////////////////////////////////
HOST_CODE RoadNetworkGraph::BaseGraph* g_hGraphCopy = 0;
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE unsigned char* g_dWorkQueuesBuffers1[NUM_PROCEDURES];
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE unsigned char* g_dWorkQueuesBuffers2[NUM_PROCEDURES];
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE MarshallingQueue g_dWorkQueues1[NUM_PROCEDURES];
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE MarshallingQueue g_dWorkQueues2[NUM_PROCEDURES];
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE unsigned char* g_dPopulationDensitiesSamplingBuffer = 0;
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE unsigned int* g_dDistancesSamplingBuffer = 0;
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE RoadNetworkGraph::Vertex* g_dVertices = 0;
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE RoadNetworkGraph::Edge* g_dEdges = 0;
//////////////////////////////////////////////////////////////////////////
HOST_CODE RoadNetworkGraph::Vertex* g_hVerticesCopy = 0;
//////////////////////////////////////////////////////////////////////////
HOST_CODE RoadNetworkGraph::Edge* g_hEdgesCopy = 0;
//////////////////////////////////////////////////////////////////////////
HOST_CODE RoadNetworkGraph::Primitive* g_hPrimitives = 0;
//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int g_hNumExtractedPrimitives = 0;
#ifdef USE_QUADTREE
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE RoadNetworkGraph::QuadTree* g_dQuadtree = 0;
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE RoadNetworkGraph::Quadrant* g_dQuadrants = 0;
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE RoadNetworkGraph::QuadrantEdges* g_dQuadrantsEdges = 0;
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE RoadNetworkGraph::EdgeIndex* g_dQueryResults = 0;
#endif
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE unsigned char* g_dPopulationDensityMapData = 0;
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE unsigned char* g_dWaterBodiesMapData = 0;
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE unsigned char* g_dBlockadesMapData = 0;
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE unsigned char* g_dNaturalPatternMapData = 0;
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE unsigned char* g_dRadialPatternMapData = 0;
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE unsigned char* g_dRasterPatternMapData = 0;
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE ImageMap* g_dPopulationDensityMap = 0;
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE ImageMap* g_dWaterBodiesMap = 0;
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE ImageMap* g_dBlockadesMap = 0;
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE ImageMap* g_dNaturalPatternMap = 0;
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE ImageMap* g_dRadialPatternMap = 0;
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE ImageMap* g_dRasterPatternMap = 0;
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE vml_vec4* g_dVerticesBuffer = 0;
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE vml_vec4* g_dColorsBuffer = 0;
//////////////////////////////////////////////////////////////////////////
DEVICE_CODE unsigned int* g_dIndicesBuffer = 0;

//////////////////////////////////////////////////////////////////////////
GLOBAL_CODE void initializeImageMap(ImageMap* imageMap, unsigned int mapWidth, unsigned int mapHeight, const unsigned char* mapData)
{
	imageMap->width = mapWidth;
	imageMap->height = mapHeight;
	imageMap->data = mapData;
}

//////////////////////////////////////////////////////////////////////////
GLOBAL_CODE void initializeWorkQueues(unsigned int capacity, unsigned int itemSize)
{
	for (unsigned int i = 0; i < NUM_PROCEDURES; i++)
	{
		g_dWorkQueues1[i].setBuffer(g_dWorkQueuesBuffers1[i], capacity);
		g_dWorkQueues1[i].setItemSize(itemSize);
		g_dWorkQueues2[i].setBuffer(g_dWorkQueuesBuffers2[i], capacity);
		g_dWorkQueues2[i].setItemSize(itemSize);
	}
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE void loadImage(const char* filePath, int desiredWidth, int desiredHeight, unsigned char* data);

//////////////////////////////////////////////////////////////////////////
HOST_CODE void allocateWorkQueues(unsigned int maxWorkQueueCapacity)
{
	freeWorkQueues();

	unsigned int capacity = maxWorkQueueCapacity;
	unsigned int itemSize = MathExtras::max(
			MathExtras::max(sizeof(Highway), sizeof(Street)), 
			MathExtras::max(sizeof(HighwayBranch), sizeof(StreetBranch))
	);
	unsigned int bufferSize = capacity * itemSize;

	for (unsigned int i = 0; i < NUM_PROCEDURES; i++)
	{
		MALLOC_ON_DEVICE(g_dWorkQueuesBuffers1[i], unsigned char, bufferSize);
		MALLOC_ON_DEVICE(g_dWorkQueuesBuffers2[i], unsigned char, bufferSize);
	}

	INVOKE_GLOBAL_CODE2(initializeWorkQueues, 1, 1, capacity, itemSize);
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE void allocateSamplingBuffers(unsigned int samplingArc)
{
	freeSamplingBuffers();

	MALLOC_ON_DEVICE(g_dPopulationDensitiesSamplingBuffer, unsigned char, samplingArc);
	MALLOC_ON_DEVICE(g_dDistancesSamplingBuffer, unsigned int, samplingArc);
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE void allocateGraphBuffers(unsigned int maxVertices, unsigned int maxEdges)
{
	freeGraphBuffers();

	MALLOC_ON_DEVICE(g_dVertices, RoadNetworkGraph::Vertex, maxVertices);
	MEMSET_ON_DEVICE(g_dVertices, 0, sizeof(RoadNetworkGraph::Vertex) * maxVertices);

	MALLOC_ON_DEVICE(g_dEdges, RoadNetworkGraph::Edge, maxEdges);
	MEMSET_ON_DEVICE(g_dEdges, 0, sizeof(RoadNetworkGraph::Edge) * maxEdges);
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE void allocateGraphicsBuffers(unsigned int verticesBufferSize, unsigned int indicesBufferSize)
{
	freeGraphicsBuffers();

	// TODO: register buffers along with OpenGL API
	MALLOC_ON_DEVICE(g_dVerticesBuffer, vml_vec4, verticesBufferSize);
	MALLOC_ON_DEVICE(g_dColorsBuffer, vml_vec4, verticesBufferSize);
	MALLOC_ON_DEVICE(g_dIndicesBuffer, unsigned int, indicesBufferSize);
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE void copyGraphBuffers(unsigned int maxVertices, unsigned int maxEdges)
{
	MEMCPY_DEVICE_TO_HOST(g_hVerticesCopy, g_dVertices, sizeof(RoadNetworkGraph::Vertex) * maxVertices);
	MEMCPY_DEVICE_TO_HOST(g_hEdgesCopy, g_dEdges, sizeof(RoadNetworkGraph::Edge) * maxEdges);
}

#ifdef USE_QUADTREE
//////////////////////////////////////////////////////////////////////////
HOST_CODE void allocateQuadtreeBuffers(unsigned int maxResultsPerQuery, unsigned int maxQuadrants)
{
	freeQuadtreeBuffers();

	MALLOC_ON_DEVICE(g_dQueryResults, RoadNetworkGraph::EdgeIndex, maxResultsPerQuery);
	MALLOC_ON_DEVICE(g_dQuadrants, RoadNetworkGraph::Quadrant, maxQuadrants);
	MALLOC_ON_DEVICE(g_dQuadrantsEdges, RoadNetworkGraph::QuadrantEdges, maxQuadrants);
	MEMSET_ON_DEVICE(g_dQuadrants, 0, sizeof(RoadNetworkGraph::Quadrant) * maxQuadrants);
	MEMSET_ON_DEVICE(g_dQuadrantsEdges, 0, sizeof(RoadNetworkGraph::QuadrantEdges) * maxQuadrants);
}
#endif

//////////////////////////////////////////////////////////////////////////
HOST_CODE void allocateAndInitializeImageMaps(const char* populationDensityMapFilePath,
									const char* waterBodiesMapFilePath,
									const char* blockadesMapFilePath,
									const char* naturalPatternMapFilePath,
									const char* radialPatternMapFilePath,
									const char* rasterPatternMapFilePath,
									unsigned int mapWidth,
									unsigned int mapHeight)
{
	freeImageMaps();

	int mapSize = mapWidth * mapHeight;

	if (strlen(populationDensityMapFilePath) > 0)
	{
		unsigned char* populationDensityMapData = (unsigned char*)malloc(sizeof(unsigned char) * mapSize);
		loadImage(populationDensityMapFilePath, mapWidth, mapHeight, populationDensityMapData);
		MALLOC_ON_DEVICE(g_dPopulationDensityMapData, unsigned char, mapSize);
		MEMCPY_HOST_TO_DEVICE(g_dPopulationDensityMapData, populationDensityMapData, sizeof(unsigned char) * mapSize);
		free(populationDensityMapData);

		MALLOC_ON_DEVICE(g_dPopulationDensityMap, ImageMap, 1);
		INVOKE_GLOBAL_CODE4(initializeImageMap, 1, 1, g_dPopulationDensityMap, mapWidth, mapHeight, g_dPopulationDensityMapData);
	}

	if (strlen(waterBodiesMapFilePath) > 0)
	{
		unsigned char* waterBodiesMapData = (unsigned char*)malloc(sizeof(unsigned char) * mapSize);
		loadImage(waterBodiesMapFilePath, mapWidth, mapHeight, waterBodiesMapData);
		MALLOC_ON_DEVICE(g_dWaterBodiesMapData, unsigned char, mapSize);
		MEMCPY_HOST_TO_DEVICE(g_dWaterBodiesMapData, waterBodiesMapData, sizeof(unsigned char) * mapSize);
		free(waterBodiesMapData);

		MALLOC_ON_DEVICE(g_dWaterBodiesMap, ImageMap, 1);
		INVOKE_GLOBAL_CODE4(initializeImageMap, 1, 1, g_dWaterBodiesMap, mapWidth, mapHeight, g_dWaterBodiesMapData);
	}

	if (strlen(blockadesMapFilePath) > 0)
	{
		unsigned char* blockadesMapData = (unsigned char*)malloc(sizeof(unsigned char) * mapSize);
		loadImage(blockadesMapFilePath, mapWidth, mapHeight, blockadesMapData);
		MALLOC_ON_DEVICE(g_dBlockadesMapData, unsigned char, mapSize);
		MEMCPY_HOST_TO_DEVICE(g_dBlockadesMapData, blockadesMapData, sizeof(unsigned char) * mapSize);
		free(blockadesMapData);

		MALLOC_ON_DEVICE(g_dBlockadesMap, ImageMap, 1);
		INVOKE_GLOBAL_CODE4(initializeImageMap, 1, 1, g_dBlockadesMap, mapWidth, mapHeight, g_dBlockadesMapData);
	}

	if (strlen(naturalPatternMapFilePath) > 0)
	{
		unsigned char* naturalPatternMapData = (unsigned char*)malloc(sizeof(unsigned char) * mapSize);
		loadImage(naturalPatternMapFilePath, mapWidth, mapHeight, naturalPatternMapData);
		MALLOC_ON_DEVICE(g_dNaturalPatternMapData, unsigned char, mapSize);
		MEMCPY_HOST_TO_DEVICE(g_dNaturalPatternMapData, naturalPatternMapData, sizeof(unsigned char) * mapSize);
		free(naturalPatternMapData);

		MALLOC_ON_DEVICE(g_dNaturalPatternMap, ImageMap, 1);
		INVOKE_GLOBAL_CODE4(initializeImageMap, 1, 1, g_dNaturalPatternMap, mapWidth, mapHeight, g_dNaturalPatternMapData);
	}

	if (strlen(radialPatternMapFilePath) > 0)
	{
		unsigned char* radialPatternMapData = (unsigned char*)malloc(sizeof(unsigned char) * mapSize);
		loadImage(radialPatternMapFilePath, mapWidth, mapHeight, radialPatternMapData);
		MALLOC_ON_DEVICE(g_dRadialPatternMapData, unsigned char, mapSize);
		MEMCPY_HOST_TO_DEVICE(g_dRadialPatternMapData, radialPatternMapData, sizeof(unsigned char) * mapSize);
		free(radialPatternMapData);

		MALLOC_ON_DEVICE(g_dRadialPatternMap, ImageMap, 1);
		INVOKE_GLOBAL_CODE4(initializeImageMap, 1, 1, g_dRadialPatternMap, mapWidth, mapHeight, g_dRadialPatternMapData);
	}

	if (strlen(rasterPatternMapFilePath) > 0)
	{
		unsigned char* rasterPatternMapData = (unsigned char*)malloc(sizeof(unsigned char) * mapSize);
		loadImage(rasterPatternMapFilePath, mapWidth, mapHeight, rasterPatternMapData);
		MALLOC_ON_DEVICE(g_dRasterPatternMapData, unsigned char, mapSize);
		MEMCPY_HOST_TO_DEVICE(g_dRasterPatternMapData, rasterPatternMapData, sizeof(unsigned char) * mapSize);
		free(rasterPatternMapData);

		MALLOC_ON_DEVICE(g_dRasterPatternMap, ImageMap, 1);
		INVOKE_GLOBAL_CODE4(initializeImageMap, 1, 1, g_dRasterPatternMap, mapWidth, mapHeight, g_dRasterPatternMapData);
	}
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE void freeWorkQueues()
{
	for (unsigned int i = 0; i < NUM_PROCEDURES; i++)
	{
		if (g_dWorkQueuesBuffers1[i] != 0)
		{
			FREE_ON_DEVICE(g_dWorkQueuesBuffers1[i]);
			g_dWorkQueuesBuffers1[i] = 0;
		}

		if (g_dWorkQueuesBuffers2[i] != 0)
		{
			FREE_ON_DEVICE(g_dWorkQueuesBuffers2[i]);
			g_dWorkQueuesBuffers2[i] = 0;
		}
	}
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE void freeSamplingBuffers()
{
	if (g_dPopulationDensitiesSamplingBuffer != 0)
	{
		FREE_ON_DEVICE(g_dPopulationDensitiesSamplingBuffer);
		g_dPopulationDensitiesSamplingBuffer = 0;
	}

	if (g_dDistancesSamplingBuffer != 0)
	{
		FREE_ON_DEVICE(g_dDistancesSamplingBuffer);
		g_dDistancesSamplingBuffer = 0;
	}
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE void freeGraphBuffers()
{
	if (g_dVertices != 0)
	{
		FREE_ON_DEVICE(g_dVertices);
		g_dVertices = 0;
	}

	if (g_dEdges != 0)
	{
		FREE_ON_DEVICE(g_dEdges);
		g_dEdges = 0;
	}
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE void freeImageMaps()
{
	if (g_dPopulationDensityMap != 0)
	{
		FREE_ON_DEVICE(g_dPopulationDensityMap);
		g_dPopulationDensityMap = 0;
	}

	if (g_dWaterBodiesMap != 0)
	{
		FREE_ON_DEVICE(g_dWaterBodiesMap);
		g_dWaterBodiesMap = 0;
	}

	if (g_dBlockadesMap != 0)
	{
		FREE_ON_DEVICE(g_dBlockadesMap);
		g_dBlockadesMap = 0;
	}

	if (g_dNaturalPatternMap != 0)
	{
		FREE_ON_DEVICE(g_dNaturalPatternMap);
		g_dNaturalPatternMap = 0;
	}

	if (g_dRadialPatternMap != 0)
	{
		FREE_ON_DEVICE(g_dRadialPatternMap);
		g_dRadialPatternMap = 0;
	}

	if (g_dRasterPatternMap != 0)
	{
		FREE_ON_DEVICE(g_dRasterPatternMap);
		g_dRasterPatternMap = 0;
	}

	if (g_dPopulationDensityMapData != 0)
	{
		FREE_ON_DEVICE(g_dPopulationDensityMapData);
		g_dPopulationDensityMapData = 0;
	}

	if (g_dWaterBodiesMapData != 0)
	{
		FREE_ON_DEVICE(g_dWaterBodiesMapData);
		g_dWaterBodiesMapData = 0;
	}

	if (g_dBlockadesMapData != 0)
	{
		FREE_ON_DEVICE(g_dBlockadesMapData);
		g_dBlockadesMapData = 0;
	}

	if (g_dNaturalPatternMapData != 0)
	{
		FREE_ON_DEVICE(g_dNaturalPatternMapData);
		g_dNaturalPatternMapData = 0;
	}

	if (g_dRadialPatternMapData != 0)
	{
		FREE_ON_DEVICE(g_dRadialPatternMapData);
		g_dRadialPatternMapData = 0;
	}

	if (g_dRasterPatternMapData != 0)
	{
		FREE_ON_DEVICE(g_dRasterPatternMapData);
		g_dRasterPatternMapData = 0;
	}
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE void freeGraphicsBuffers()
{
	if (g_dVerticesBuffer != 0)
	{
		// TODO:
		FREE_ON_DEVICE(g_dVerticesBuffer);
		g_dVerticesBuffer = 0;
	}

	if (g_dColorsBuffer != 0)
	{
		// TODO:
		FREE_ON_DEVICE(g_dColorsBuffer);
		g_dColorsBuffer = 0;
	}

	if (g_dIndicesBuffer != 0)
	{
		// TODO:
		FREE_ON_DEVICE(g_dIndicesBuffer);
		g_dIndicesBuffer = 0;
	}
}

#ifdef USE_QUADTREE
//////////////////////////////////////////////////////////////////////////
HOST_CODE void freeQuadtreeBuffers()
{
	if (g_dQuadrants != 0)
	{
		FREE_ON_DEVICE(g_dQuadrants);
		g_dQuadrants = 0;
	}

	if (g_dQuadrantsEdges != 0)
	{
		FREE_ON_DEVICE(g_dQuadrantsEdges);
		g_dQuadrantsEdges = 0;
	}

	if (g_dQueryResults != 0)
	{
		FREE_ON_DEVICE(g_dQueryResults);
		g_dQueryResults = 0;
	}
}
#endif

//////////////////////////////////////////////////////////////////////////
HOST_CODE void loadImage(const char* filePath, int width, int height, unsigned char* data)
{
	FREE_IMAGE_FORMAT format = FreeImage_GetFileType(filePath, 0);
	FIBITMAP* bitmap = FreeImage_Load(format, filePath);
	FIBITMAP* image = FreeImage_ConvertTo32Bits(bitmap);
	int imageWidth = FreeImage_GetWidth(image);
	int imageHeight = FreeImage_GetHeight(image);

	if (imageWidth != width || imageHeight != height)
	{
		image = FreeImage_Rescale(image, width, height, FILTER_BOX);
	}

	int size = width * height;
	unsigned char* bgra = (unsigned char*)FreeImage_GetBits(image);

	if (bgra == 0)
	{
		throw std::exception("invalid image file");
	}

	for (int i = 0, j = 0; i < size; i++, j += 4)
	{
		// grayscale = (0.21 R + 0.71 G + 0.07 B)
		data[i] = (unsigned char)(0.21f * bgra[j + 2] + 0.71f * bgra[j + 1] + 0.07f * bgra[j]);
	}

	FreeImage_Unload(image);
}