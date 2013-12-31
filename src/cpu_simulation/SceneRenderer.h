#ifndef SCENERENDERER_H
#define SCENERENDERER_H

#include <Renderer.h>
#include <Configuration.h>
#include <Camera.h>
#include <Shader.h>
#include <RoadNetworkGeometry.h>
#include <Quad.h>

#include <GL3/gl3w.h>
#include <GL/utils/gl.h>

#include <vector>

struct ImageMapRenderData
{
	Texture* texture;
	vml_vec4 color1;
	vml_vec4 color2;
	bool enabled;

	ImageMapRenderData() : texture(0), enabled(false)
	{
	}

};

class SceneRenderer : public Renderer
{
public:
	SceneRenderer(Camera& camera, RoadNetworkGeometry& roadNetworkGeometry) :
		Renderer(camera),
		solidShader("../../../../../shaders/solid.vs.glsl", "../../../../../shaders/solid.fs.glsl"),
		imageMapShader("../../../../../shaders/imageMap.vs.glsl", "../../../../../shaders/imageMap.fs.glsl"),
		roadNetworkGeometry(roadNetworkGeometry),
		worldSizedQuad(0)
	{
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glEnable(GL_PROGRAM_POINT_SIZE);
		glPointSize(3.0f);
		glEnable(GL_CULL_FACE);
		glCullFace(GL_BACK);
	}

	~SceneRenderer()
	{
		if (worldSizedQuad != 0)
		{
			delete worldSizedQuad;
		}

		destroyImageMaps();
	}

	void setUpImageMaps(const Box2D& worldBounds, const ImageMap* populationDensityMap, const ImageMap* waterBodiesMap, const ImageMap* blockadesMap)
	{
		destroyImageMaps();

		if ((populationDensityMapData.enabled = (populationDensityMap != 0)))
		{
			setUpImageMapRenderData(populationDensityMap, populationDensityMapData, BLACK_COLOR, WHITE_COLOR);
		}

		if ((waterBodiesMapData.enabled = (waterBodiesMap != 0)))
		{
			setUpImageMapRenderData(waterBodiesMap, waterBodiesMapData, vml_vec4(0.0f, 0.0f, 0.0f, 0.0f), WATER_COLOR);
		}

		if ((blockadesMapData.enabled = (blockadesMap != 0)))
		{
			setUpImageMapRenderData(blockadesMap, blockadesMapData, vml_vec4(0.0f, 0.0f, 0.0f, 0.0f), GRASS_COLOR);
		}
		
		vml_vec2 size = worldBounds.getExtents();

		if (worldSizedQuad != 0)
		{
			if (worldSizedQuad->getX() != worldBounds.min.x || worldSizedQuad->getY() != worldBounds.min.y ||
				worldSizedQuad->getWidth() != size.x || worldSizedQuad->getHeight() != size.y)
			{
				delete worldSizedQuad;
			}

			else
			{
				return;
			}
		}

		worldSizedQuad = new Quad(worldBounds.min.x, worldBounds.min.y, size.x, size.y);
	}

	virtual void render(double deltaTime)
	{
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		vml_mat4 viewProjection = camera.getProjectionMatrix() * camera.getViewMatrix();

		if (worldSizedQuad != 0)
		{
			// render image maps
			glDepthMask(GL_FALSE);
			glEnable(GL_BLEND);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
			imageMapShader.bind();
			imageMapShader.setMat4("uViewProjection", viewProjection);

			if (populationDensityMapData.enabled && populationDensityMapData.texture != 0)
			{
				drawImageMap(populationDensityMapData);
			}

			if (waterBodiesMapData.enabled && waterBodiesMapData.texture != 0)
			{
				drawImageMap(waterBodiesMapData);
			}

			if (blockadesMapData.enabled && blockadesMapData.texture != 0)
			{
				drawImageMap(blockadesMapData);
			}

			imageMapShader.unbind();
			glDisable(GL_BLEND);
			glDepthMask(GL_TRUE);
		}

		// render road network
		solidShader.bind();
		solidShader.setMat4("uViewProjection", viewProjection);
		roadNetworkGeometry.draw();
		solidShader.unbind();
	}

	void togglePopulationDensityMap()
	{
		populationDensityMapData.enabled = !populationDensityMapData.enabled;
	}

	void toggleWaterBodiesMap()
	{
		waterBodiesMapData.enabled = !waterBodiesMapData.enabled;
	}

	void toggleBlockadesMap()
	{
		blockadesMapData.enabled = !blockadesMapData.enabled;
	}

private:
	Shader solidShader;
	Shader imageMapShader;
	RoadNetworkGeometry& roadNetworkGeometry;
	Quad* worldSizedQuad;
	bool drawPopulationDensityMap;
	bool drawWaterBodiesMap;
	bool drawBlockadesMap;
	ImageMapRenderData populationDensityMapData;
	ImageMapRenderData waterBodiesMapData;
	ImageMapRenderData blockadesMapData;

	void setUpImageMapRenderData(const ImageMap* imageMap, ImageMapRenderData& imageMapData, const vml_vec4& color1, const vml_vec4& color2)
	{
		imageMapData.texture = new Texture(imageMap->width, imageMap->height, GL_RED, GL_R8, GL_UNSIGNED_BYTE, GL_NEAREST, GL_CLAMP_TO_EDGE, (void*)imageMap->data);
		imageMapData.color1 = color1;
		imageMapData.color2 = color2;
	}

	void destroyImageMaps()
	{
		if (populationDensityMapData.texture != 0)
		{
			delete populationDensityMapData.texture;
		}

		if (waterBodiesMapData.texture != 0)
		{
			delete waterBodiesMapData.texture;
		}

		if (blockadesMapData.texture != 0)
		{
			delete blockadesMapData.texture;
		}
	}

	void drawImageMap(const ImageMapRenderData& imageMapRenderData)
	{
		imageMapShader.setTexture("uBaseTex", *imageMapRenderData.texture, 0);
		imageMapShader.setVec4("uColor1", imageMapRenderData.color1);
		imageMapShader.setVec4("uColor2", imageMapRenderData.color2);
		worldSizedQuad->draw();
	}


};

#endif