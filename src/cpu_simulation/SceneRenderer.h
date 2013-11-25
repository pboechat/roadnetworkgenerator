#ifndef SCENERENDERER_H
#define SceneRENDERER_H

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

struct ImageMapShadingData
{
	Texture* texture;
	glm::vec4 color1;
	glm::vec4 color2;

};

class SceneRenderer : public Renderer
{
public:
	SceneRenderer(Camera& camera, unsigned int screenWidth, unsigned int screenHeight, RoadNetworkGeometry& roadNetworkGeometry, const ImageMap& populationDensityMap, const ImageMap& waterBodiesMap) :
		Renderer(camera),
		solidShader("../../../../../shaders/solid.vs.glsl", "../../../../../shaders/solid.fs.glsl"),
		imageMapShader("../../../../../shaders/imageMap.vs.glsl", "../../../../../shaders/imageMap.fs.glsl"),
		screenSizedQuad(glm::vec3(0.0f, 0.0f, 0.0f), (float)screenWidth, (float)screenHeight),
		roadNetworkGeometry(roadNetworkGeometry)
	{
		glClearColor(0, 0, 0, 1);
		glEnable(GL_PROGRAM_POINT_SIZE);
		glPointSize(3.0f);
		glEnable(GL_CULL_FACE);
		glCullFace(GL_BACK);
		createImageMapShadingData(populationDensityMap);
		createImageMapShadingData(waterBodiesMap);
	}

	~SceneRenderer()
	{
		for (unsigned int i = 0; i < imageMaps.size(); i++)
		{
			delete imageMaps[i].texture;
		}
		imageMaps.clear();
	}

	virtual void render(double deltaTime)
	{
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glm::mat4 viewProjection = camera.getProjectionMatrix() * camera.getViewMatrix();
		// render image maps
		glDepthMask(GL_FALSE);
		glEnable(GL_BLEND);
		glBlendFunc(GL_ONE, GL_ONE);
		imageMapShader.bind();
		imageMapShader.setMat4("uViewProjection", viewProjection);

		for (unsigned int i = 0; i < imageMaps.size(); i++)
		{
			ImageMapShadingData& imageMap = imageMaps[i];
			imageMapShader.setTexture("uBaseTex", *imageMap.texture, 0);
			imageMapShader.setVec4("uColor1", imageMap.color1);
			imageMapShader.setVec4("uColor2", imageMap.color2);
			screenSizedQuad.draw();
		}

		imageMapShader.unbind();
		glDisable(GL_BLEND);
		glDepthMask(GL_TRUE);
		// render road network
		solidShader.bind();
		solidShader.setMat4("uViewProjection", viewProjection);
		roadNetworkGeometry.draw();
		solidShader.unbind();
	}

private:
	Shader solidShader;
	Shader imageMapShader;
	RoadNetworkGeometry& roadNetworkGeometry;
	Quad screenSizedQuad;
	std::vector<ImageMapShadingData> imageMaps;

	void createImageMapShadingData(const ImageMap& imageMap)
	{
		ImageMapShadingData shadingData;
		shadingData.texture = new Texture(imageMap.getWidth(), imageMap.getHeight(), GL_RED, GL_R8, GL_UNSIGNED_BYTE, GL_NEAREST, GL_CLAMP_TO_EDGE, (void*)imageMap.getData());
		shadingData.color1 = imageMap.getColor1();
		shadingData.color2 = imageMap.getColor2();
		imageMaps.push_back(shadingData);
	}

};

#endif