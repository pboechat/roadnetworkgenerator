#ifndef ROADNETWORKGEOMETRY_H
#define ROADNETWORKGEOMETRY_H

#include <Geometry.h>
#include <Configuration.h>
#include <Line.h>
#include <AABB.h>

#include <GL3/gl3w.h>

#include <vector>

class RoadNetworkGeometry : public Geometry
{
public:
	AABB bounds;

	RoadNetworkGeometry() : elementsCount(0)
	{
		glGenBuffers(2, buffers);
		glGenVertexArrays(1, &vao);
	}

	~RoadNetworkGeometry()
	{
		glDeleteBuffers(2, buffers);
		glDeleteVertexArrays(1, &vao);
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	void build(const Configuration& configuration, const std::vector<Line>& lines)
	{
		elementsCount = lines.size() * 2;
		glm::vec4* vertices = new glm::vec4[elementsCount];
		unsigned int* indices = new unsigned int[elementsCount];
		bounds.min = glm::vec3(10000, 10000, 10000);
		bounds.max = glm::vec3(-10000, -10000, -10000);

		for (unsigned int i = 0, j = 0; i < lines.size(); i++, j += 2)
		{
			const Line& segment = lines[i];
			glm::vec3 v1(segment.start.x, segment.start.y, segment.start.z);
			glm::vec3 v2(segment.end.x, segment.end.y, segment.end.z);
			vertices[j] = glm::vec4(v1.x, v1.y, v1.z, 1.0f);
			vertices[j + 1] = glm::vec4(v2.x, v2.y, v2.z, 1.0f);
			indices[j] = j;
			indices[j + 1] = j + 1;
			bounds.min = glm::min(glm::vec3(v1), bounds.min);
			bounds.max = glm::max(glm::vec3(v1), bounds.max);
			bounds.min = glm::min(glm::vec3(v2), bounds.min);
			bounds.max = glm::max(glm::vec3(v2), bounds.max);
		}

		glBindBuffer(GL_ARRAY_BUFFER, buffers[0]);
		glBufferData(GL_ARRAY_BUFFER, elementsCount * sizeof(glm::vec4), (void*)vertices, GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		delete[] vertices;
		
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffers[1]);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, elementsCount * sizeof(unsigned int), (void*)indices, GL_STATIC_DRAW);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		delete[] indices;

		glBindVertexArray(vao);
		glBindBuffer(GL_ARRAY_BUFFER, buffers[0]);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);
		glBindVertexArray(0);
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	virtual void draw()
	{
		if (elementsCount == 0)
		{
			return;
		}

		glBindVertexArray(vao);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffers[1]);
		glDrawElements(GL_LINES, elementsCount, GL_UNSIGNED_INT, 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		glBindVertexArray(0);
	}

private:
	unsigned int buffers[2];
	unsigned int vao;
	unsigned int elementsCount;

};

#endif