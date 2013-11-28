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
#include <glm/glm.hpp>

#include <vector>

struct ImageMapRenderData
{
	Texture* texture;
	glm::vec4 color1;
	glm::vec4 color2;

	ImageMapRenderData() : texture(0)
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
		drawPopulationDensityMap(true),
		drawWaterBodiesMap(true),
		worldSizedQuad(0)
	{
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glEnable(GL_PROGRAM_POINT_SIZE);
		glPointSize(2.0f);
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

	void setUpImageMaps(const AABB& worldBounds, const ImageMap& populationDensityMap, const ImageMap& waterBodiesMap)
	{
		destroyImageMaps();
		setUpImageMapRenderData(populationDensityMap, populationDensityMapData);
		setUpImageMapRenderData(waterBodiesMap, waterBodiesMapData);
		glm::vec3 size = worldBounds.getExtents();

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
		glm::mat4 viewProjection = camera.getProjectionMatrix() * camera.getViewMatrix();

		if (worldSizedQuad != 0)
		{
			// render image maps
			glDepthMask(GL_FALSE);
			glEnable(GL_BLEND);
			glBlendFunc(GL_ONE, GL_ONE);
			imageMapShader.bind();
			imageMapShader.setMat4("uViewProjection", viewProjection);
			if (drawPopulationDensityMap)
			{
				drawImageMap(populationDensityMapData);
			}
			if (drawWaterBodiesMap)
			{
				drawImageMap(waterBodiesMapData);
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
		drawPopulationDensityMap = !drawPopulationDensityMap;
	}

	void toggleWaterBodiesMap() 
	{
		drawWaterBodiesMap = !drawWaterBodiesMap;
	}

private:
	Shader solidShader;
	Shader imageMapShader;
	RoadNetworkGeometry& roadNetworkGeometry;
	Quad* worldSizedQuad;
	bool drawPopulationDensityMap;
	bool drawWaterBodiesMap;
	ImageMapRenderData populationDensityMapData;
	ImageMapRenderData waterBodiesMapData;

	void setUpImageMapRenderData(const ImageMap& imageMap, ImageMapRenderData& imageMapData)
	{
		imageMapData.texture = new Texture(imageMap.getWidth(), imageMap.getHeight(), GL_RED, GL_R8, GL_UNSIGNED_BYTE, GL_NEAREST, GL_CLAMP_TO_EDGE, (void*)imageMap.getData());
		imageMapData.color1 = imageMap.getColor1();
		imageMapData.color2 = imageMap.getColor2();
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