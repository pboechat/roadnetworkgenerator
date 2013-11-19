#ifndef IMAGEMAPQUAD_H
#define IMAGEMAPQUAD_H

#include <Geometry.h>
#include <Texture.h>
#include <ImageMap.h>

#include <glm/glm.hpp>

class ImageMapQuad : public Geometry
{
public:
	ImageMapQuad() : texture(0) 
	{
		glGenBuffers(3, buffers);
		glGenVertexArrays(1, &vao);
	}

	~ImageMapQuad() 
	{ 
		if (texture != 0) 
		{
			delete texture; 
		}

		glDeleteBuffers(3, buffers);
		glDeleteVertexArrays(1, &vao);
	}

	void build(ImageMap& imageMap)
	{
		int width = imageMap.getWidth();
		int height = imageMap.getHeight();

		texture = new Texture(width, height, GL_RED, GL_R8, GL_UNSIGNED_BYTE, GL_NEAREST, GL_CLAMP_TO_EDGE, (void*)imageMap.getData());

		glm::vec4 vertices[4];
		vertices[0] = glm::vec4((float)width, (float)height, 0.0f, 1.0f);
		vertices[1] = glm::vec4(0.0f, (float)height, 0.0f, 1.0f);
		vertices[2] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
		vertices[3] = glm::vec4((float)width, 0.0f, 0.0f, 1.0f);

		glm::vec2 uvs[4];
		uvs[0] = glm::vec2(1.0f, 1.0f);
		uvs[1] = glm::vec2(0.0f, 1.0f);
		uvs[2] = glm::vec2(0.0f, 0.0f);
		uvs[3] = glm::vec2(1.0f, 0.0f);

		unsigned int indices[6];
		indices[0] = 0;
		indices[1] = 1;
		indices[2] = 2;
		indices[3] = 0;
		indices[4] = 2;
		indices[5] = 3;

		glBindBuffer(GL_ARRAY_BUFFER, buffers[0]);
		glBufferData(GL_ARRAY_BUFFER, 4 * sizeof(glm::vec4), (void*)vertices, GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glBindBuffer(GL_ARRAY_BUFFER, buffers[1]);
		glBufferData(GL_ARRAY_BUFFER, 4 * sizeof(glm::vec2), (void*)uvs, GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffers[2]);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, 6 * sizeof(unsigned int), (void*)indices, GL_STATIC_DRAW);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

		glBindVertexArray(vao);
		glBindBuffer(GL_ARRAY_BUFFER, buffers[0]);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindBuffer(GL_ARRAY_BUFFER, buffers[1]);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, 0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindVertexArray(0);
	}

	virtual void draw() 
	{
		if (texture == 0)
		{
			return;
		}

		glBindVertexArray(vao);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffers[2]);
		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		glBindVertexArray(0);
	}

	inline Texture* getTexture()
	{
		return texture;
	}

private:
	unsigned int buffers[3];
	unsigned int vao;
	Texture* texture;

};

#endif