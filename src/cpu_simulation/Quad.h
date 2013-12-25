#ifndef QUAD_H
#define QUAD_H

#include <Geometry.h>

#include <glm/glm.hpp>

class Quad : public Geometry
{
public:
	Quad(float x, float y, float width, float height) : x(x), y(y), width(width), height(height)
	{
		glGenBuffers(3, buffers);
		glGenVertexArrays(1, &vao);
		glm::vec4 vertices[4];
		vertices[0] = glm::vec4(x + width, y + height, 0.0f, 1.0f);
		vertices[1] = glm::vec4(x, y + height, 0.0f, 1.0f);
		vertices[2] = glm::vec4(x, y, 0.0f, 1.0f);
		vertices[3] = glm::vec4(x + width, y, 0.0f, 1.0f);
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

	~Quad()
	{
		glDeleteBuffers(3, buffers);
		glDeleteVertexArrays(1, &vao);
	}

	inline float getX() const
	{
		return x;
	}

	inline float getY() const
	{
		return y;
	}

	inline float getWidth() const
	{
		return width;
	}

	inline float getHeight() const
	{
		return width;
	}

	virtual void draw()
	{
		glBindVertexArray(vao);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffers[2]);
		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		glBindVertexArray(0);
	}

private:
	float x;
	float y;
	float width;
	float height;
	unsigned int buffers[3];
	unsigned int vao;

};

#endif