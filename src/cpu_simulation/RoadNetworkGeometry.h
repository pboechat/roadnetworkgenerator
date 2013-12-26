#ifndef ROADNETWORKGEOMETRY_H
#define ROADNETWORKGEOMETRY_H

#include <Geometry.h>
#include <Configuration.h>
#include <GraphTraversal.h>

#include <vector_math.h>
#include <GL3/gl3w.h>

#include <vector>

struct GeometryCreationTraversal : public RoadNetworkGraph::GraphTraversal
{
	std::vector<vml_vec4>& vertices;
	std::vector<vml_vec4>& colors;
	std::vector<unsigned int>& indices;
	vml_vec4 highwayColor;
	vml_vec4 streetColor;

	GeometryCreationTraversal(std::vector<vml_vec4>& vertices, std::vector<vml_vec4>& colors, std::vector<unsigned int>& indices, const vml_vec4& highwayColor, const vml_vec4& streetColor) : vertices(vertices), colors(colors), indices(indices), highwayColor(highwayColor), streetColor(streetColor) {}
	~GeometryCreationTraversal() {}

	virtual bool operator () (const vml_vec2& source, const vml_vec2& destination, bool highway)
	{
		unsigned int i = vertices.size();
		vertices.push_back(vml_vec4(source.x, source.y, 0.0f, 1.0f));
		vertices.push_back(vml_vec4(destination.x, destination.y, 0.0f, 1.0f));
		vml_vec4 color = (highway) ? highwayColor : streetColor;
		colors.push_back(color);
		colors.push_back(color);
		indices.push_back(i);
		indices.push_back(i + 1);
		return true;
	}

};

class RoadNetworkGeometry : public Geometry
{
public:
	RoadNetworkGeometry() : built(false), elementsCount(0)
	{
	}

	~RoadNetworkGeometry()
	{
		if (built)
		{
			glDeleteBuffers(3, buffers);
			glDeleteVertexArrays(1, &vao);
		}
	}

	void build(const RoadNetworkGraph::Graph& roadNetworkGraph, const vml_vec4& highwayColor, const vml_vec4& streetColor)
	{
		if (!built)
		{
			glGenBuffers(3, buffers);
			glGenVertexArrays(1, &vao);
		}

		std::vector<vml_vec4> vertices;
		std::vector<vml_vec4> colors;
		std::vector<unsigned int> indices;
		roadNetworkGraph.traverse(GeometryCreationTraversal(vertices, colors, indices, highwayColor, streetColor));
		glBindBuffer(GL_ARRAY_BUFFER, buffers[0]);
		glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(vml_vec4), (void*)&vertices[0], GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindBuffer(GL_ARRAY_BUFFER, buffers[1]);
		glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(vml_vec4), (void*)&colors[0], GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		elementsCount = indices.size();
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffers[2]);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), (void*)&indices[0], GL_STATIC_DRAW);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
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
		built = true;
	}

	virtual void draw()
	{
		if (!built)
		{
			return;
		}

		glBindVertexArray(vao);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffers[2]);
		glDrawElements(GL_LINES, elementsCount, GL_UNSIGNED_INT, 0);
		glDrawElements(GL_POINTS, elementsCount, GL_UNSIGNED_INT, 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		glBindVertexArray(0);
	}

private:
	unsigned int buffers[3];
	unsigned int vao;
	unsigned int elementsCount;
	bool built;

};

#endif