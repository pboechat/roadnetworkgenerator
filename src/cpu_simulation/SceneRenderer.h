#ifndef SCENERENDERER_H
#define SceneRENDERER_H

#include <Renderer.h>
#include <Camera.h>
#include <Shader.h>
#include <RoadNetworkGeometry.h>
#include <ImageMapQuad.h>

#include <GL3/gl3w.h>
#include <GL/utils/gl.h>
#include <glm/glm.hpp>

#include <vector>

class SceneRenderer : public Renderer
{
public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	SceneRenderer(Camera& camera, RoadNetworkGeometry& roadNetworkGeometry, std::vector<ImageMapQuad*>& imageMapQuads) : 
		Renderer(camera), 
		solidShader("../../../../../shaders/solid.vs.glsl", "../../../../../shaders/solid.fs.glsl"), 
		imageMapShader("../../../../../shaders/imageMap.vs.glsl", "../../../../../shaders/imageMap.fs.glsl"), 
		roadNetworkGeometry(roadNetworkGeometry), 
		imageMapQuads(imageMapQuads)
	{
		glClearColor(0, 0, 0, 1);
		glEnable(GL_CULL_FACE);
		glCullFace(GL_BACK);
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	virtual void render(double deltaTime)
	{
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glm::mat4 viewProjection = camera.getProjectionMatrix() * camera.getViewMatrix();

		glDepthMask(GL_FALSE);
		glEnable(GL_BLEND);
		glBlendFunc(GL_ONE, GL_ONE);

		imageMapShader.bind();
		imageMapShader.setMat4("uViewProjection", viewProjection);
		for (unsigned int i = 0; i < imageMapQuads.size(); i++) 
		{
			ImageMapQuad* imageMapQuad = imageMapQuads[i];
			imageMapShader.setTexture("uBaseTex", *imageMapQuad->getTexture(), 0);
			imageMapQuad->draw();
		}
		imageMapShader.unbind();

		glDisable(GL_BLEND);
		glDepthMask(GL_TRUE);

		solidShader.bind();
		solidShader.setMat4("uViewProjection", viewProjection);
		roadNetworkGeometry.draw();
		solidShader.unbind();
	}

private:
	Shader solidShader;
	Shader imageMapShader;
	RoadNetworkGeometry& roadNetworkGeometry;
	std::vector<ImageMapQuad*>& imageMapQuads;

};

#endif