#ifndef ROADNETWORKGEOMETRY_H
#define ROADNETWORKGEOMETRY_H

#include <Geometry.h>
#include <Configuration.h>
#include <Line.h>
#include <AABB.h>
#include <QuadTree.h>

#include <GL3/gl3w.h>

#include <vector>

#define _DRAW_QUADTREE

class RoadNetworkGeometry : public Geometry
{
public:
	AABB bounds;

	RoadNetworkGeometry() : elementsCount(0)
	{
		glGenBuffers(3, buffers);
		glGenVertexArrays(1, &vao);
	}

	~RoadNetworkGeometry()
	{
		glDeleteBuffers(3, buffers);
		glDeleteVertexArrays(1, &vao);
	}

#ifdef _DRAW_QUADTREE
	void addLeafQuadrantsBounds(const QuadTree& quadtree, std::vector<AABB>& quadrantsBounds) 
	{
		if (quadtree.isLeaf())
		{
			if (quadtree.hasLines()) 
			{
				quadrantsBounds.push_back(quadtree.getBounds());
			}
		} 
		else 
		{
			addLeafQuadrantsBounds(*quadtree.getNorthWest(), quadrantsBounds);
			addLeafQuadrantsBounds(*quadtree.getNorthEast(), quadrantsBounds);
			addLeafQuadrantsBounds(*quadtree.getSouthWest(), quadrantsBounds);
			addLeafQuadrantsBounds(*quadtree.getSouthEast(), quadrantsBounds);
		}
	}

	void build(const Configuration& configuration, const std::vector<Line>& lines, const QuadTree& quadtree)
	{
		std::vector<AABB> quadrantsBounds;
		addLeafQuadrantsBounds(quadtree, quadrantsBounds);

		unsigned int verticesCount = lines.size() * 2 + quadrantsBounds.size() * 4;
		elementsCount = lines.size() * 2 + quadrantsBounds.size() * 8;

		glm::vec4* vertices = new glm::vec4[verticesCount];
		glm::vec4* colors = new glm::vec4[verticesCount];
		unsigned int* indices = new unsigned int[elementsCount];

		bounds.min = glm::vec3(10000, 10000, 10000);
		bounds.max = glm::vec3(-10000, -10000, -10000);
		unsigned int j = 0;
		for (unsigned int i = 0; i < lines.size(); i++, j += 2)
		{
			const Line& segment = lines[i];

			glm::vec3 v1(segment.start.x, segment.start.y, segment.start.z);
			glm::vec3 v2(segment.end.x, segment.end.y, segment.end.z);

			vertices[j] = glm::vec4(v1.x, v1.y, v1.z, 1.0f);
			vertices[j + 1] = glm::vec4(v2.x, v2.y, v2.z, 1.0f);

			colors[j] = segment.color1;
			colors[j + 1] = segment.color2;

			indices[j] = j;
			indices[j + 1] = j + 1;

			bounds.min = glm::min(glm::vec3(v1), bounds.min);
			bounds.max = glm::max(glm::vec3(v1), bounds.max);
			bounds.min = glm::min(glm::vec3(v2), bounds.min);
			bounds.max = glm::max(glm::vec3(v2), bounds.max);
		}

		for (unsigned int i = 0, k = j; i < quadrantsBounds.size(); i++, j += 4, k += 8)
		{
			AABB& quadrantBound = quadrantsBounds[i];

			vertices[j] = glm::vec4(quadrantBound.min.x, quadrantBound.max.y, 0.0f, 1.0f);
			vertices[j + 1] = glm::vec4(quadrantBound.max.x, quadrantBound.max.y, 0.0f, 1.0f);
			vertices[j + 2] = glm::vec4(quadrantBound.max.x, quadrantBound.min.y, 0.0f, 1.0f);
			vertices[j + 3] = glm::vec4(quadrantBound.min.x, quadrantBound.min.y, 0.0f, 1.0f);

			colors[j] = configuration.quadtreeColor;
			colors[j + 1] = configuration.quadtreeColor;
			colors[j + 2] = configuration.quadtreeColor;
			colors[j + 3] = configuration.quadtreeColor;

			indices[k] = j;
			indices[k + 1] = j + 1;

			indices[k + 2] = j + 1;
			indices[k + 3] = j + 2;

			indices[k + 4] = j + 2;
			indices[k + 5] = j + 3;

			indices[k + 6] = j + 3;
			indices[k + 7] = j;
		}

		glBindBuffer(GL_ARRAY_BUFFER, buffers[0]);
		glBufferData(GL_ARRAY_BUFFER, verticesCount * sizeof(glm::vec4), (void*)vertices, GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		delete[] vertices;

		glBindBuffer(GL_ARRAY_BUFFER, buffers[1]);
		glBufferData(GL_ARRAY_BUFFER, verticesCount * sizeof(glm::vec4), (void*)colors, GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		delete[] colors;

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffers[2]);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, elementsCount * sizeof(unsigned int), (void*)indices, GL_STATIC_DRAW);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		delete[] indices;

		glBindVertexArray(vao);
		glBindBuffer(GL_ARRAY_BUFFER, buffers[0]);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindBuffer(GL_ARRAY_BUFFER, buffers[1]);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, 0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindVertexArray(0);
	}
#else
	void build(const Configuration& configuration, const std::vector<Line>& lines)
	{
		elementsCount = lines.size() * 2;

		glm::vec4* vertices = new glm::vec4[elementsCount];
		glm::vec4* colors = new glm::vec4[elementsCount];
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
			colors[j] = segment.color1;
			colors[j + 1] = segment.color2;
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

		glBindBuffer(GL_ARRAY_BUFFER, buffers[1]);
		glBufferData(GL_ARRAY_BUFFER, elementsCount * sizeof(glm::vec4), (void*)colors, GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		delete[] colors;
		
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffers[2]);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, elementsCount * sizeof(unsigned int), (void*)indices, GL_STATIC_DRAW);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		delete[] indices;

		glBindVertexArray(vao);
		glBindBuffer(GL_ARRAY_BUFFER, buffers[0]);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindBuffer(GL_ARRAY_BUFFER, buffers[1]);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, 0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindVertexArray(0);
	}
#endif

	virtual void draw()
	{
		if (elementsCount == 0)
		{
			return;
		}

		glBindVertexArray(vao);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffers[2]);
		glDrawElements(GL_LINES, elementsCount, GL_UNSIGNED_INT, 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		glBindVertexArray(0);
	}

private:
	unsigned int buffers[3];
	unsigned int vao;
	unsigned int elementsCount;

};

#endif