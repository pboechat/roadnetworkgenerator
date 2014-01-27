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
HOST_CODE RoadNetworkGraph::Graph* g_hGraph = 0;
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
HOST_CODE RoadNetworkGraph::Vertex* g_hVertices = 0;
//////////////////////////////////////////////////////////////////////////
HOST_CODE RoadNetworkGraph::Edge* g_hEdges = 0;
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
HOST_CODE RoadNetworkGraph::QuadTree* g_hQuadtree = 0;
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
HOST_CODE unsigned char* g_hPopulationDensityMapData = 0;
//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned char* g_hWaterBodiesMapData = 0;
//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned char* g_hBlockadesMapData = 0;
//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned char* g_hNaturalPatternMapData = 0;
//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned char* g_hRadialPatternMapData = 0;
//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned char* g_hRasterPatternMapData = 0;
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
HOST_CODE ImageMap* g_hPopulationDensityMap = 0;
//////////////////////////////////////////////////////////////////////////
HOST_CODE ImageMap* g_hWaterBodiesMap = 0;
//////////////////////////////////////////////////////////////////////////
HOST_CODE ImageMap* g_hBlockadesMap = 0;
//////////////////////////////////////////////////////////////////////////
HOST_CODE ImageMap* g_hNaturalPatternMap = 0;
//////////////////////////////////////////////////////////////////////////
HOST_CODE ImageMap* g_hRadialPatternMap = 0;
//////////////////////////////////////////////////////////////////////////
HOST_CODE ImageMap* g_hRasterPatternMap = 0;
//////////////////////////////////////////////////////////////////////////
HOST_CODE vml_vec4* g_hVerticesBuffer = 0;
//////////////////////////////////////////////////////////////////////////
HOST_CODE vml_vec4* g_hColorsBuffer = 0;
//////////////////////////////////////////////////////////////////////////
HOST_CODE unsigned int* g_hIndicesBuffer = 0;

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
HOST_CODE void allocateGraph(unsigned int maxVertices, unsigned int maxEdges)
{
	freeGraph();

	g_hGraph = (RoadNetworkGraph::Graph*)malloc(sizeof(RoadNetworkGraph::Graph));
	g_hGraphCopy = (RoadNetworkGraph::BaseGraph*)malloc(sizeof(RoadNetworkGraph::BaseGraph));
	g_hVertices = (RoadNetworkGraph::Vertex*)malloc(sizeof(RoadNetworkGraph::Vertex) * g_hConfiguration->maxVertices);
	g_hEdges = (RoadNetworkGraph::Edge*)malloc(sizeof(RoadNetworkGraph::Edge) * g_hConfiguration->maxEdges);
	g_hVerticesCopy = (RoadNetworkGraph::Vertex*)malloc(sizeof(RoadNetworkGraph::Vertex) * g_hConfiguration->maxVertices);
	g_hEdgesCopy = (RoadNetworkGraph::Edge*)malloc(sizeof(RoadNetworkGraph::Edge) * g_hConfiguration->maxEdges);

	MALLOC_ON_DEVICE(g_dGraph, RoadNetworkGraph::Graph, 1);

	MALLOC_ON_DEVICE(g_dVertices, RoadNetworkGraph::Vertex, maxVertices);
	MEMSET_ON_DEVICE(g_dVertices, 0, sizeof(RoadNetworkGraph::Vertex) * maxVertices);

	MALLOC_ON_DEVICE(g_dEdges, RoadNetworkGraph::Edge, maxEdges);
	MEMSET_ON_DEVICE(g_dEdges, 0, sizeof(RoadNetworkGraph::Edge) * maxEdges);
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE void copyGraphToHost(unsigned int maxVertices, unsigned int maxEdges)
{
	MEMCPY_DEVICE_TO_HOST(g_hVertices, g_dVertices, sizeof(RoadNetworkGraph::Vertex) * maxVertices);
	MEMCPY_DEVICE_TO_HOST(g_hEdges, g_dEdges, sizeof(RoadNetworkGraph::Edge) * maxEdges);

	MEMCPY_DEVICE_TO_HOST(g_hGraph, g_dGraph, sizeof(RoadNetworkGraph::Graph));
	g_hGraph->vertices = g_hVertices;
	g_hGraph->edges = g_hEdges;

	memcpy(g_hVerticesCopy, g_hVertices, sizeof(RoadNetworkGraph::Vertex) * maxVertices);
	memcpy(g_hEdgesCopy, g_hEdges, sizeof(RoadNetworkGraph::Edge) * maxEdges);
	g_hGraphCopy->vertices = g_hVerticesCopy;
	g_hGraphCopy->edges = g_hEdgesCopy;

	RoadNetworkGraph::copy(g_hGraph, g_hGraphCopy);
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE void copyGraphToDevice(unsigned int maxVertices, unsigned int maxEdges)
{
	MEMCPY_HOST_TO_DEVICE(g_dVertices, g_hVertices, sizeof(RoadNetworkGraph::Vertex) * maxVertices);
	MEMCPY_HOST_TO_DEVICE(g_dEdges, g_hEdges, sizeof(RoadNetworkGraph::Edge) * maxEdges);
	INVOKE_GLOBAL_CODE3(RoadNetworkGraph::updateNumVerticesAndNumEdges, 1, 1, g_dGraph, g_hGraph->numVertices, g_hGraph->numEdges);
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE void copyQuadtreeToHost()
{
	MEMCPY_DEVICE_TO_HOST(g_hQuadtree, g_dQuadtree, sizeof(RoadNetworkGraph::QuadTree));
}

#ifdef USE_QUADTREE
//////////////////////////////////////////////////////////////////////////
HOST_CODE void allocateQuadtree(unsigned int maxResultsPerQuery, unsigned int maxQuadrants)
{
	freeQuadtree();

	g_hQuadtree = (RoadNetworkGraph::QuadTree*)malloc(sizeof(RoadNetworkGraph::QuadTree));

	MALLOC_ON_DEVICE(g_dQuadtree, RoadNetworkGraph::QuadTree, 1);
	MALLOC_ON_DEVICE(g_dQueryResults, RoadNetworkGraph::EdgeIndex, maxResultsPerQuery);
	MALLOC_ON_DEVICE(g_dQuadrants, RoadNetworkGraph::Quadrant, maxQuadrants);
	MALLOC_ON_DEVICE(g_dQuadrantsEdges, RoadNetworkGraph::QuadrantEdges, maxQuadrants);
	MEMSET_ON_DEVICE(g_dQuadrants, 0, sizeof(RoadNetworkGraph::Quadrant) * maxQuadrants);
	MEMSET_ON_DEVICE(g_dQuadrantsEdges, 0, sizeof(RoadNetworkGraph::QuadrantEdges) * maxQuadrants);
}
#endif

//////////////////////////////////////////////////////////////////////////
HOST_CODE void allocateAndInitializeConfiguration(const std::string& configurationFile)
{
	freeConfiguration();

	g_hConfiguration = (Configuration*)malloc(sizeof(Configuration));
	g_hConfiguration->loadFromFile(configurationFile);

	MALLOC_ON_DEVICE(g_dConfiguration, Configuration, 1);
	MEMCPY_HOST_TO_DEVICE(g_dConfiguration, g_hConfiguration, sizeof(Configuration));
}

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
		g_hPopulationDensityMapData = (unsigned char*)malloc(sizeof(unsigned char) * mapSize);
		loadImage(populationDensityMapFilePath, mapWidth, mapHeight, g_hPopulationDensityMapData);
		MALLOC_ON_DEVICE(g_dPopulationDensityMapData, unsigned char, mapSize);
		MEMCPY_HOST_TO_DEVICE(g_dPopulationDensityMapData, g_hPopulationDensityMapData, sizeof(unsigned char) * mapSize);

		g_hPopulationDensityMap = (ImageMap*)malloc(sizeof(ImageMap));
		initializeImageMap(g_hPopulationDensityMap, mapWidth, mapHeight, g_hPopulationDensityMapData);
		MALLOC_ON_DEVICE(g_dPopulationDensityMap, ImageMap, 1);
		INVOKE_GLOBAL_CODE4(initializeImageMap, 1, 1, g_dPopulationDensityMap, mapWidth, mapHeight, g_dPopulationDensityMapData);
	}

	if (strlen(waterBodiesMapFilePath) > 0)
	{
		g_hWaterBodiesMapData = (unsigned char*)malloc(sizeof(unsigned char) * mapSize);
		loadImage(waterBodiesMapFilePath, mapWidth, mapHeight, g_hWaterBodiesMapData);
		MALLOC_ON_DEVICE(g_dWaterBodiesMapData, unsigned char, mapSize);
		MEMCPY_HOST_TO_DEVICE(g_dWaterBodiesMapData, g_hWaterBodiesMapData, sizeof(unsigned char) * mapSize);

		g_hWaterBodiesMap = (ImageMap*)malloc(sizeof(ImageMap));
		initializeImageMap(g_hWaterBodiesMap, mapWidth, mapHeight, g_hWaterBodiesMapData);
		MALLOC_ON_DEVICE(g_dWaterBodiesMap, ImageMap, 1);
		INVOKE_GLOBAL_CODE4(initializeImageMap, 1, 1, g_dWaterBodiesMap, mapWidth, mapHeight, g_dWaterBodiesMapData);
	}

	if (strlen(blockadesMapFilePath) > 0)
	{
		g_hBlockadesMapData = (unsigned char*)malloc(sizeof(unsigned char) * mapSize);
		loadImage(blockadesMapFilePath, mapWidth, mapHeight, g_hBlockadesMapData);
		MALLOC_ON_DEVICE(g_dBlockadesMapData, unsigned char, mapSize);
		MEMCPY_HOST_TO_DEVICE(g_dBlockadesMapData, g_hBlockadesMapData, sizeof(unsigned char) * mapSize);

		g_hBlockadesMap = (ImageMap*)malloc(sizeof(ImageMap));
		initializeImageMap(g_hBlockadesMap, mapWidth, mapHeight, g_hBlockadesMapData);
		MALLOC_ON_DEVICE(g_dBlockadesMap, ImageMap, 1);
		INVOKE_GLOBAL_CODE4(initializeImageMap, 1, 1, g_dBlockadesMap, mapWidth, mapHeight, g_dBlockadesMapData);
	}

	if (strlen(naturalPatternMapFilePath) > 0)
	{
		g_hNaturalPatternMapData = (unsigned char*)malloc(sizeof(unsigned char) * mapSize);
		loadImage(naturalPatternMapFilePath, mapWidth, mapHeight, g_hNaturalPatternMapData);
		MALLOC_ON_DEVICE(g_dNaturalPatternMapData, unsigned char, mapSize);
		MEMCPY_HOST_TO_DEVICE(g_dNaturalPatternMapData, g_hNaturalPatternMapData, sizeof(unsigned char) * mapSize);

		g_hNaturalPatternMap = (ImageMap*)malloc(sizeof(ImageMap));
		initializeImageMap(g_hNaturalPatternMap, mapWidth, mapHeight, g_hNaturalPatternMapData);
		MALLOC_ON_DEVICE(g_dNaturalPatternMap, ImageMap, 1);
		INVOKE_GLOBAL_CODE4(initializeImageMap, 1, 1, g_dNaturalPatternMap, mapWidth, mapHeight, g_dNaturalPatternMapData);
	}

	if (strlen(radialPatternMapFilePath) > 0)
	{
		g_hRadialPatternMapData = (unsigned char*)malloc(sizeof(unsigned char) * mapSize);
		loadImage(radialPatternMapFilePath, mapWidth, mapHeight, g_hRadialPatternMapData);
		MALLOC_ON_DEVICE(g_dRadialPatternMapData, unsigned char, mapSize);
		MEMCPY_HOST_TO_DEVICE(g_dRadialPatternMapData, g_hRadialPatternMapData, sizeof(unsigned char) * mapSize);

		g_hRadialPatternMap = (ImageMap*)malloc(sizeof(ImageMap));
		initializeImageMap(g_hRadialPatternMap, mapWidth, mapHeight, g_hRadialPatternMapData);
		MALLOC_ON_DEVICE(g_dRadialPatternMap, ImageMap, 1);
		INVOKE_GLOBAL_CODE4(initializeImageMap, 1, 1, g_dRadialPatternMap, mapWidth, mapHeight, g_dRadialPatternMapData);
	}

	if (strlen(rasterPatternMapFilePath) > 0)
	{
		g_hRasterPatternMapData = (unsigned char*)malloc(sizeof(unsigned char) * mapSize);
		loadImage(rasterPatternMapFilePath, mapWidth, mapHeight, g_hRasterPatternMapData);
		MALLOC_ON_DEVICE(g_dRasterPatternMapData, unsigned char, mapSize);
		MEMCPY_HOST_TO_DEVICE(g_dRasterPatternMapData, g_hRasterPatternMapData, sizeof(unsigned char) * mapSize);

		g_hRasterPatternMap = (ImageMap*)malloc(sizeof(ImageMap));
		initializeImageMap(g_hRasterPatternMap, mapWidth, mapHeight, g_hRasterPatternMapData);
		MALLOC_ON_DEVICE(g_dRasterPatternMap, ImageMap, 1);
		INVOKE_GLOBAL_CODE4(initializeImageMap, 1, 1, g_dRasterPatternMap, mapWidth, mapHeight, g_dRasterPatternMapData);
	}
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE void allocatePrimitives()
{
	freePrimitives();

	g_hPrimitives = (RoadNetworkGraph::Primitive*)malloc(sizeof(RoadNetworkGraph::Primitive) * g_hConfiguration->maxPrimitives);
	memset(g_hPrimitives, 0, sizeof(RoadNetworkGraph::Primitive) * g_hConfiguration->maxPrimitives);
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE void allocateGraphicsBuffers(unsigned int vertexBufferSize, unsigned int indexBufferSize)
{
	if (g_hVerticesBuffer == 0)
	{
		g_hVerticesBuffer = (vml_vec4*)malloc(sizeof(vml_vec4) * vertexBufferSize);
	}

	if (g_hColorsBuffer == 0)
	{
		g_hColorsBuffer = (vml_vec4*)malloc(sizeof(vml_vec4) * vertexBufferSize);
	}

	if (g_hIndicesBuffer == 0)
	{
		g_hIndicesBuffer = (unsigned int*)malloc(sizeof(unsigned int) * indexBufferSize);
	}
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE void freeConfiguration()
{
	if (g_hConfiguration != 0)
	{
		free(g_hConfiguration);
		g_hConfiguration = 0;
	}

	if (g_dConfiguration != 0)
	{
		FREE_ON_DEVICE(g_dConfiguration);
		g_dConfiguration = 0;
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
HOST_CODE void freeGraph()
{
	if (g_hGraph != 0)
	{
		free(g_hGraph);
		g_hGraph = 0;
	}

	if (g_hGraphCopy != 0)
	{
		free(g_hGraphCopy);
		g_hGraphCopy = 0;
	}

	if (g_hVertices != 0)
	{
		free(g_hVertices);
		g_hVertices = 0;
	}

	if (g_hEdges != 0)
	{
		free(g_hEdges);
		g_hEdges = 0;
	}

	if (g_hVerticesCopy != 0)
	{
		free(g_hVerticesCopy);
		g_hVerticesCopy = 0;
	}

	if (g_hEdgesCopy != 0)
	{
		free(g_hEdgesCopy);
		g_hEdgesCopy = 0;
	}

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

	if (g_dGraph != 0)
	{
		FREE_ON_DEVICE(g_dGraph);
		g_dGraph = 0;
	}
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE void freeImageMaps()
{
	if (g_hPopulationDensityMap != 0)
	{
		free(g_hPopulationDensityMap);
		g_hPopulationDensityMap = 0;
	}

	if (g_hWaterBodiesMap != 0)
	{
		free(g_hWaterBodiesMap);
		g_hWaterBodiesMap = 0;
	}

	if (g_hBlockadesMap != 0)
	{
		free(g_hBlockadesMap);
		g_hBlockadesMap = 0;
	}

	if (g_hNaturalPatternMap != 0)
	{
		free(g_hNaturalPatternMap);
		g_hNaturalPatternMap = 0;
	}

	if (g_hRadialPatternMap != 0)
	{
		free(g_hRadialPatternMap);
		g_hRadialPatternMap = 0;
	}

	if (g_hRasterPatternMap != 0)
	{
		free(g_hRasterPatternMap);
		g_hRasterPatternMap = 0;
	}

	if (g_hPopulationDensityMapData != 0)
	{
		free(g_hPopulationDensityMapData);
		g_hPopulationDensityMapData = 0;
	}

	if (g_hWaterBodiesMapData != 0)
	{
		free(g_hWaterBodiesMapData);
		g_hWaterBodiesMapData = 0;
	}

	if (g_hBlockadesMapData != 0)
	{
		free(g_hBlockadesMapData);
		g_hBlockadesMapData = 0;
	}

	if (g_hNaturalPatternMapData != 0)
	{
		free(g_hNaturalPatternMapData);
		g_hNaturalPatternMapData = 0;
	}

	if (g_hRadialPatternMapData != 0)
	{
		free(g_hRadialPatternMapData);
		g_hRadialPatternMapData = 0;
	}

	if (g_hRasterPatternMapData != 0)
	{
		free(g_hRasterPatternMapData);
		g_hRasterPatternMapData = 0;
	}

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

#ifdef USE_QUADTREE
//////////////////////////////////////////////////////////////////////////
HOST_CODE void freeQuadtree()
{
	if (g_hQuadtree != 0)
	{
		free(g_hQuadtree);
		g_hQuadtree = 0;
	}

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

	if (g_dQuadtree != 0)
	{
		FREE_ON_DEVICE(g_dQuadtree);
		g_dQuadtree = 0;
	}
}
#endif

//////////////////////////////////////////////////////////////////////////
HOST_CODE void freePrimitives()
{
	if (g_hPrimitives != 0)
	{
		free(g_hPrimitives);
		g_hPrimitives = 0;
	}
}

//////////////////////////////////////////////////////////////////////////
HOST_CODE void freeGraphicsBuffers()
{
	if (g_hVerticesBuffer != 0)
	{
		free(g_hVerticesBuffer);
		g_hVerticesBuffer = 0;
	}

	if (g_hColorsBuffer != 0)
	{
		free(g_hColorsBuffer);
		g_hColorsBuffer = 0;
	}

	if (g_hIndicesBuffer != 0)
	{
		free(g_hIndicesBuffer);
		g_hIndicesBuffer = 0;
	}
}

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