#ifndef ROADNETWORKRENDERER_H
#define ROADNETWORKRENDERER_H

#include <Renderer.h>
#include <Camera.h>
#include <Shader.h>
#include <RoadNetworkGeometry.h>

#include <GL3/gl3w.h>
#include <GL/utils/gl.h>
#include <glm/glm.hpp>

class SceneRenderer : public Renderer
{
public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	SceneRenderer(Camera& camera, RoadNetworkGeometry& geometry)
		: Renderer(camera), solidShader("../../../../../shaders/roadnetwork.vs.glsl", "../../../../../shaders/roadnetwork.fs.glsl"), geometry(geometry)
	{
		glClearColor(0, 0, 0, 1);
		glEnable(GL_CULL_FACE);
		glCullFace(GL_BACK);
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	virtual void render(double deltaTime)
	{
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glm::mat4 view = camera.getViewMatrix();
		glm::mat4 viewInverse = glm::inverse(view);
		glm::mat4 projection = camera.getProjectionMatrix();
		glm::mat4 viewProjection = projection * view;
		solidShader.bind();
		solidShader.setMat4("uView", view);
		solidShader.setMat4("uViewInverse", viewInverse);
		solidShader.setMat4("uViewInverseTranspose", glm::transpose(viewInverse));
		solidShader.setMat4("uViewProjection", viewProjection);
		solidShader.setVec4("uLightDir", -glm::normalize(glm::vec4(0.0f, -1.0f, -1.0f, 1.0f) * viewInverse));
		solidShader.setFloat("uLightIntensity", 0.8f);
		solidShader.setVec4("uLightColor", glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));
		solidShader.setVec4("uAmbientColor", glm::vec4(0.1f, 0.1f, 0.1f, 1.0f));
		solidShader.setVec4("uDiffuseColor", glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));
		solidShader.setVec4("uSpecularColor", glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));
		solidShader.setFloat("uShininess", 30.0f);
		roadNetworkGeometry.draw();
		GL_CHECK_ERROR();
		solidShader.unbind();
	}

private:
	Shader solidShader;
	RoadNetworkGeometry roadNetworkGeometry;

};

#endif