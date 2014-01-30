#ifndef RENDERER_H
#define RENDERER_H

#pragma once

#include <Camera.h>

#include <GL3/gl3w.h>

#include <exception>

class Renderer
{
public:
	////////////////////////////////////////////////////////////////////////////////////////////////////
	void initialize()
	{
		glEnable(GL_TEXTURE_2D);
		glEnable(GL_DEPTH_TEST);
		glEnable(GL_CULL_FACE);
		glCullFace(GL_BACK);
		glViewport(0, 0, camera.getScreenWidth(), camera.getScreenHeight());
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	virtual void render(double elapsedTime) = 0;

protected:
	Camera& camera;

	Renderer(Camera& camera) : camera(camera) {}

};

#endif