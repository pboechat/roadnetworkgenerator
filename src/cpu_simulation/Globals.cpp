#include <Globals.h>
#include <Road.h>
#include <Branch.h>

#include <FreeImage.h>

#include <memory>
#include <exception>

//////////////////////////////////////////////////////////////////////////
Configuration* g_configuration = 0;
//////////////////////////////////////////////////////////////////////////
RoadNetworkGraph::Graph* g_graph = 0;
//////////////////////////////////////////////////////////////////////////
RoadNetworkGraph::BaseGraph* g_graphCopy = 0;
//////////////////////////////////////////////////////////////////////////
unsigned char* g_workQueuesBuffers1[NUM_PROCEDURES];
//////////////////////////////////////////////////////////////////////////
unsigned char* g_workQueuesBuffers2[NUM_PROCEDURES];
//////////////////////////////////////////////////////////////////////////
StaticMarshallingQueue g_workQueues1[NUM_PROCEDURES];
//////////////////////////////////////////////////////////////////////////
StaticMarshallingQueue g_workQueues2[NUM_PROCEDURES];
//////////////////////////////////////////////////////////////////////////
unsigned char* g_populationDensitiesSamplingBuffer = 0;
//////////////////////////////////////////////////////////////////////////
unsigned int* g_distancesSamplingBuffer = 0;
//////////////////////////////////////////////////////////////////////////
RoadNetworkGraph::Vertex* g_vertices = 0;
//////////////////////////////////////////////////////////////////////////
RoadNetworkGraph::Edge* g_edges = 0;
//////////////////////////////////////////////////////////////////////////
RoadNetworkGraph::Vertex* g_verticesCopy = 0;
//////////////////////////////////////////////////////////////////////////
RoadNetworkGraph::Edge* g_edgesCopy = 0;
//////////////////////////////////////////////////////////////////////////
RoadNetworkGraph::Primitive* g_primitives = 0;
//////////////////////////////////////////////////////////////////////////
unsigned int g_numExtractedPrimitives = 0;
#ifdef USE_QUADTREE
//////////////////////////////////////////////////////////////////////////
extern RoadNetworkGraph::QuadTree* g_quadtree = 0;
//////////////////////////////////////////////////////////////////////////
RoadNetworkGraph::Quadrant* g_quadrants = 0;
//////////////////////////////////////////////////////////////////////////
RoadNetworkGraph::QuadrantEdges* g_quadrantsEdges = 0;
//////////////////////////////////////////////////////////////////////////
RoadNetworkGraph::EdgeIndex* g_queryResults = 0;
#endif
//////////////////////////////////////////////////////////////////////////
unsigned char* g_populationDensityMapData = 0;
//////////////////////////////////////////////////////////////////////////
unsigned char* g_waterBodiesMapData = 0;
//////////////////////////////////////////////////////////////////////////
unsigned char* g_blockadesMapData = 0;
//////////////////////////////////////////////////////////////////////////
unsigned char* g_naturalPatternMapData = 0;
//////////////////////////////////////////////////////////////////////////
unsigned char* g_radialPatternMapData = 0;
//////////////////////////////////////////////////////////////////////////
unsigned char* g_rasterPatternMapData = 0;
//////////////////////////////////////////////////////////////////////////
ImageMap* g_populationDensityMap = 0;
//////////////////////////////////////////////////////////////////////////
ImageMap* g_waterBodiesMap = 0;
//////////////////////////////////////////////////////////////////////////
ImageMap* g_blockadesMap = 0;
//////////////////////////////////////////////////////////////////////////
ImageMap* g_naturalPatternMap = 0;
//////////////////////////////////////////////////////////////////////////
ImageMap* g_radialPatternMap = 0;
//////////////////////////////////////////////////////////////////////////
ImageMap* g_rasterPatternMap = 0;
//////////////////////////////////////////////////////////////////////////
vml_vec4* g_verticesBuffer = 0;
//////////////////////////////////////////////////////////////////////////
vml_vec4* g_colorsBuffer = 0;
//////////////////////////////////////////////////////////////////////////
unsigned int* g_indicesBuffer = 0;

//////////////////////////////////////////////////////////////////////////
void loadImage(const char* filePath, int desiredWidth, int desiredHeight, unsigned char* data);
//////////////////////////////////////////////////////////////////////////
void initializeImageMap(ImageMap* imageMap, unsigned int mapWidth, unsigned int mapHeight, const unsigned char* mapData);

//////////////////////////////////////////////////////////////////////////
void initializeWorkQueues()
{
	for (unsigned int i = 0; i < NUM_PROCEDURES; i++)
	{
		g_workQueuesBuffers1[i] = 0;
		g_workQueuesBuffers2[i] = 0;
	}
}

//////////////////////////////////////////////////////////////////////////
void allocateWorkQueues(unsigned int maxWorkQueueCapacity)
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
		g_workQueuesBuffers1[i] = (unsigned char*)malloc(sizeof(unsigned char) * bufferSize);

		if (g_workQueuesBuffers1[i] == 0)
		{
			throw std::exception("g_workQueuesBuffers1: insufficient memory");
		}

		g_workQueuesBuffers2[i] = (unsigned char*)malloc(sizeof(unsigned char) * bufferSize);

		if (g_workQueuesBuffers2[i] == 0)
		{
			throw std::exception("g_workQueuesBuffers2: insufficient memory");
		}

		g_workQueues1[i].setBuffer(g_workQueuesBuffers1[i], capacity);
		g_workQueues1[i].setItemSize(itemSize);
		g_workQueues2[i].setBuffer(g_workQueuesBuffers2[i], capacity);
		g_workQueues2[i].setItemSize(itemSize);
	}
}

//////////////////////////////////////////////////////////////////////////
void freeWorkQueues()
{
	for (unsigned int i = 0; i < NUM_PROCEDURES; i++)
	{
		if (g_workQueuesBuffers1[i] != 0)
		{
			free(g_workQueuesBuffers1[i]);
			g_workQueuesBuffers1[i] = 0;
		}

		if (g_workQueuesBuffers2[i] != 0)
		{
			free(g_workQueuesBuffers2[i]);
			g_workQueuesBuffers2[i] = 0;
		}
	}
}

//////////////////////////////////////////////////////////////////////////
void allocateSamplingBuffers(unsigned int samplingArc)
{
	freeSamplingBuffers();
	g_populationDensitiesSamplingBuffer = (unsigned char*)malloc(sizeof(unsigned char) * samplingArc);

	if (g_populationDensitiesSamplingBuffer == 0)
	{
		throw std::exception("g_populationDensitiesSamplingBuffer: insufficient memory");
	}

	g_distancesSamplingBuffer = (unsigned int*)malloc(sizeof(unsigned int) * samplingArc);

	if (g_distancesSamplingBuffer == 0)
	{
		throw std::exception("g_distancesSamplingBuffer: insufficient memory");
	}
}

//////////////////////////////////////////////////////////////////////////
void freeSamplingBuffers()
{
	if (g_populationDensitiesSamplingBuffer != 0)
	{
		free(g_populationDensitiesSamplingBuffer);
		g_populationDensitiesSamplingBuffer = 0;
	}

	if (g_distancesSamplingBuffer != 0)
	{
		free(g_distancesSamplingBuffer);
		g_distancesSamplingBuffer = 0;
	}
}

//////////////////////////////////////////////////////////////////////////
void allocateGraphBuffers(unsigned int maxVertices, unsigned int maxEdges)
{
	freeGraphBuffers();
	g_vertices = (RoadNetworkGraph::Vertex*)malloc(sizeof(RoadNetworkGraph::Vertex) * maxVertices);

	if (g_vertices == 0)
	{
		throw std::exception("g_vertices: insufficient memory");
	}

	memset(g_vertices, 0, sizeof(RoadNetworkGraph::Vertex) * maxVertices);

	g_verticesCopy = (RoadNetworkGraph::Vertex*)malloc(sizeof(RoadNetworkGraph::Vertex) * maxVertices);

	if (g_verticesCopy == 0)
	{
		throw std::exception("g_verticesCopy: insufficient memory");
	}

	g_edges = (RoadNetworkGraph::Edge*)malloc(sizeof(RoadNetworkGraph::Edge) * maxEdges);

	if (g_edges == 0)
	{
		throw std::exception("g_edges: insufficient memory");
	}

	memset(g_edges, 0, sizeof(RoadNetworkGraph::Edge) * maxEdges);

	g_edgesCopy = (RoadNetworkGraph::Edge*)malloc(sizeof(RoadNetworkGraph::Edge) * maxEdges);

	if (g_edgesCopy == 0)
	{
		throw std::exception("g_edgesCopy: insufficient memory");
	}
}

//////////////////////////////////////////////////////////////////////////
void allocateGraphicsBuffers(unsigned int verticesBufferSize, unsigned int indicesBufferSize)
{
	freeGraphicsBuffers();

	g_verticesBuffer = (vml_vec4*)malloc(sizeof(vml_vec4) * verticesBufferSize);

	if (g_verticesBuffer == 0)
	{
		throw std::exception("g_verticesBuffer: insufficient memory");
	}

	g_colorsBuffer = (vml_vec4*)malloc(sizeof(vml_vec4) * verticesBufferSize);

	if (g_colorsBuffer == 0)
	{
		throw std::exception("g_colorsBuffer: insufficient memory");
	}

	g_indicesBuffer = (unsigned int*)malloc(sizeof(unsigned int) * indicesBufferSize);

	if (g_indicesBuffer == 0)
	{
		throw std::exception("g_indicesBuffer: insufficient memory");
	}
}

//////////////////////////////////////////////////////////////////////////
void copyGraphBuffers(unsigned int maxVertices, unsigned int maxEdges)
{
	memcpy(g_verticesCopy, g_vertices, sizeof(RoadNetworkGraph::Vertex) * maxVertices);
	memcpy(g_edgesCopy, g_edges, sizeof(RoadNetworkGraph::Edge) * maxEdges);
}

//////////////////////////////////////////////////////////////////////////
void allocatePrimitivesBuffer(unsigned int maxPrimitives)
{
	freePrimitivesBuffer();
	g_primitives = (RoadNetworkGraph::Primitive*)malloc(sizeof(RoadNetworkGraph::Primitive) * maxPrimitives);

	if (g_primitives == 0)
	{
		throw std::exception("g_primitives: insufficient memory");
	}

	memset(g_primitives, 0, sizeof(RoadNetworkGraph::Primitive) * maxPrimitives);
}

//////////////////////////////////////////////////////////////////////////
void freeGraphBuffers()
{
	if (g_vertices != 0)
	{
		free(g_vertices);
		g_vertices = 0;
	}

	if (g_edges != 0)
	{
		free(g_edges);
		g_edges = 0;
	}

	if (g_verticesCopy != 0)
	{
		free(g_verticesCopy);
		g_verticesCopy = 0;
	}

	if (g_edgesCopy != 0)
	{
		free(g_edgesCopy);
		g_edgesCopy = 0;
	}
}

//////////////////////////////////////////////////////////////////////////
void freePrimitivesBuffer()
{
	if (g_primitives != 0)
	{
		free(g_primitives);
		g_primitives = 0;
	}
}

#ifdef USE_QUADTREE
//////////////////////////////////////////////////////////////////////////
void allocateQuadtreeBuffers(unsigned int maxResultsPerQuery, unsigned int maxQuadrants)
{
	freeQuadtreeBuffers();
	g_queryResults = (RoadNetworkGraph::EdgeIndex*)malloc(sizeof(RoadNetworkGraph::EdgeIndex) * maxResultsPerQuery);

	if (g_queryResults == 0)
	{
		throw std::exception("g_queryResults: insufficient memory");
	}

	g_quadrants = (RoadNetworkGraph::Quadrant*)malloc(sizeof(RoadNetworkGraph::Quadrant) * maxQuadrants);

	if (g_quadrants == 0)
	{
		throw std::exception("g_quadrants: insufficient memory");
	}

	g_quadrantsEdges = (RoadNetworkGraph::QuadrantEdges*)malloc(sizeof(RoadNetworkGraph::QuadrantEdges) * maxQuadrants);

	if (g_quadrantsEdges == 0)
	{
		throw std::exception("g_quadrantsEdges: insufficient memory");
	}

	memset(g_quadrants, 0, sizeof(RoadNetworkGraph::Quadrant) * maxQuadrants);
	memset(g_quadrantsEdges, 0, sizeof(RoadNetworkGraph::QuadrantEdges) * maxQuadrants);
}

//////////////////////////////////////////////////////////////////////////
void freeQuadtreeBuffers()
{
	if (g_quadrants != 0)
	{
		free(g_quadrants);
	}

	if (g_quadrantsEdges != 0)
	{
		free(g_quadrantsEdges);
	}

	if (g_queryResults != 0)
	{
		free(g_queryResults);
	}
}
#endif

//////////////////////////////////////////////////////////////////////////
void allocateAndInitializeImageMaps(const char* populationDensityMapFilePath,
									const char* waterBodiesMapFilePath,
									const char* blockadesMapFilePath,
									const char* naturalPatternMapFilePath,
									const char* radialPatternMapFileMap,
									const char* rasterPatternMapFileMap,
									unsigned int mapWidth,
									unsigned int mapHeight)
{
	freeImageMaps();
	int mapSize = mapWidth * mapHeight;

	if (strlen(populationDensityMapFilePath) > 0)
	{
		g_populationDensityMapData = (unsigned char*)malloc(sizeof(unsigned char) * mapSize);
		loadImage(populationDensityMapFilePath, mapWidth, mapHeight, g_populationDensityMapData);
		g_populationDensityMap = (ImageMap*)malloc(sizeof(ImageMap));
		initializeImageMap(g_populationDensityMap, mapWidth, mapHeight, g_populationDensityMapData);
	}

	if (strlen(waterBodiesMapFilePath) > 0)
	{
		g_waterBodiesMapData = (unsigned char*)malloc(sizeof(unsigned char) * mapSize);
		loadImage(waterBodiesMapFilePath, mapWidth, mapHeight, g_waterBodiesMapData);
		g_waterBodiesMap = (ImageMap*)malloc(sizeof(ImageMap));
		initializeImageMap(g_waterBodiesMap, mapWidth, mapHeight, g_waterBodiesMapData);
	}

	if (strlen(blockadesMapFilePath) > 0)
	{
		g_blockadesMapData = (unsigned char*)malloc(sizeof(unsigned char) * mapSize);
		loadImage(blockadesMapFilePath, mapWidth, mapHeight, g_blockadesMapData);
		g_blockadesMap = (ImageMap*)malloc(sizeof(ImageMap));
		initializeImageMap(g_blockadesMap, mapWidth, mapHeight, g_blockadesMapData);
	}

	if (strlen(naturalPatternMapFilePath) > 0)
	{
		g_naturalPatternMapData = (unsigned char*)malloc(sizeof(unsigned char) * mapSize);
		loadImage(naturalPatternMapFilePath, mapWidth, mapHeight, g_naturalPatternMapData);
		g_naturalPatternMap = (ImageMap*)malloc(sizeof(ImageMap));
		initializeImageMap(g_naturalPatternMap, mapWidth, mapHeight, g_naturalPatternMapData);
	}

	if (strlen(radialPatternMapFileMap) > 0)
	{
		g_radialPatternMapData = (unsigned char*)malloc(sizeof(unsigned char) * mapSize);
		loadImage(radialPatternMapFileMap, mapWidth, mapHeight, g_radialPatternMapData);
		g_radialPatternMap = (ImageMap*)malloc(sizeof(ImageMap));
		initializeImageMap(g_radialPatternMap, mapWidth, mapHeight, g_radialPatternMapData);
	}

	if (strlen(rasterPatternMapFileMap) > 0)
	{
		g_rasterPatternMapData = (unsigned char*)malloc(sizeof(unsigned char) * mapSize);
		loadImage(rasterPatternMapFileMap, mapWidth, mapHeight, g_rasterPatternMapData);
		g_rasterPatternMap = (ImageMap*)malloc(sizeof(ImageMap));
		initializeImageMap(g_rasterPatternMap, mapWidth, mapHeight, g_rasterPatternMapData);
	}
}

//////////////////////////////////////////////////////////////////////////
void freeImageMaps()
{
	if (g_populationDensityMap != 0)
	{
		delete g_populationDensityMap;
		g_populationDensityMap = 0;
	}

	if (g_waterBodiesMap != 0)
	{
		delete g_waterBodiesMap;
		g_waterBodiesMap = 0;
	}

	if (g_blockadesMap != 0)
	{
		delete g_blockadesMap;
		g_blockadesMap = 0;
	}

	if (g_naturalPatternMap != 0)
	{
		delete g_naturalPatternMap;
		g_naturalPatternMap = 0;
	}

	if (g_radialPatternMap != 0)
	{
		delete g_radialPatternMap;
		g_radialPatternMap = 0;
	}

	if (g_rasterPatternMap != 0)
	{
		delete g_rasterPatternMap;
		g_rasterPatternMap = 0;
	}

	if (g_populationDensityMapData != 0)
	{
		delete[] g_populationDensityMapData;
		g_populationDensityMapData = 0;
	}

	if (g_waterBodiesMapData != 0)
	{
		delete[] g_waterBodiesMapData;
		g_waterBodiesMapData = 0;
	}

	if (g_blockadesMapData != 0)
	{
		delete[] g_blockadesMapData;
		g_blockadesMapData = 0;
	}

	if (g_naturalPatternMapData != 0)
	{
		delete[] g_naturalPatternMapData;
		g_naturalPatternMapData = 0;
	}

	if (g_radialPatternMapData != 0)
	{
		delete[] g_radialPatternMapData;
		g_radialPatternMapData = 0;
	}

	if (g_rasterPatternMapData != 0)
	{
		delete[] g_rasterPatternMapData;
		g_rasterPatternMapData = 0;
	}
}

//////////////////////////////////////////////////////////////////////////
void freeGraphicsBuffers()
{
	if (g_verticesBuffer != 0)
	{
		delete[] g_verticesBuffer;
		g_verticesBuffer = 0;
	}

	if (g_colorsBuffer != 0)
	{
		delete[] g_colorsBuffer;
		g_colorsBuffer = 0;
	}

	if (g_indicesBuffer != 0)
	{
		delete[] g_indicesBuffer;
		g_indicesBuffer = 0;
	}
}

//////////////////////////////////////////////////////////////////////////
void initializeImageMap(ImageMap* imageMap, unsigned int mapWidth, unsigned int mapHeight, const unsigned char* mapData)
{
	imageMap->width = mapWidth;
	imageMap->height = mapHeight;
	imageMap->data = mapData;
}

//////////////////////////////////////////////////////////////////////////
void loadImage(const char* filePath, int width, int height, unsigned char* data)
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