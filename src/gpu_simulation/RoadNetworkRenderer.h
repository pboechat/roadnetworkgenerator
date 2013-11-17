#ifndef ROADNETWORKRENDERER_H
#define ROADNETWORKRENDERER_H

#include <Renderer.h>
#include <Camera.h>
#include <Shader.h>
#include <RoadNetworkGeometry.h>

#include <GL3/gl3w.h>
#include <GL/utils/gl.h>
#include <glm/glm.hpp>

class RoadNetworkRenderer : public Renderer
{
public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	RoadNetworkRenderer(Camera& camera, RoadNetworkGeometry& geometry)
		: Renderer(camera), shader("../../../../../shaders/roadnetwork.vs.glsl", "../../../../../shaders/roadnetwork.fs.glsl"), geometry(geometry)
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
		shader.bind();
		shader.setMat4("uView", view);
		shader.setMat4("uViewInverse", viewInverse);
		shader.setMat4("uViewInverseTranspose", glm::transpose(viewInverse));
		shader.setMat4("uViewProjection", viewProjection);
		shader.setVec4("uLightDir", -glm::normalize(glm::vec4(0.0f, -1.0f, -1.0f, 1.0f) * viewInverse));
		shader.setFloat("uLightIntensity", 0.8f);
		shader.setVec4("uLightColor", glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));
		shader.setVec4("uAmbientColor", glm::vec4(0.1f, 0.1f, 0.1f, 1.0f));
		shader.setVec4("uDiffuseColor", glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));
		shader.setVec4("uSpecularColor", glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));
		shader.setFloat("uShininess", 30.0f);
		geometry.draw();
		GL_CHECK_ERROR();
		shader.unbind();
	}

private:
	Shader shader;
	RoadNetworkGeometry geometry;

};

#endif