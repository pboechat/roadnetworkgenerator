#ifndef ROADNETWORKGEOMETRY_H
#define ROADNETWORKGEOMETRY_H

#include <Geometry.h>
#include <Globals.h>
#include <Primitive.h>
#include <GraphTraversal.h>

#include <vector_math.h>
#include <GL3/gl3w.h>

#include <vector>

class RoadNetworkGeometry : public Geometry
{
private:
	struct GeometryCreationTraversal : public RoadNetworkGraph::GraphTraversal
	{
		std::vector<vml_vec4>& vertices;
		std::vector<vml_vec4>& colors;
		std::vector<unsigned int>& indices;
		vml_vec4 color;

		GeometryCreationTraversal(std::vector<vml_vec4>& vertices, std::vector<vml_vec4>& colors, std::vector<unsigned int>& indices, const vml_vec4& color) : vertices(vertices), colors(colors), indices(indices), color(color) {}
		~GeometryCreationTraversal() {}

		virtual bool operator () (const RoadNetworkGraph::Vertex& source, const RoadNetworkGraph::Vertex& destination, const RoadNetworkGraph::Edge& edge)
		{
			if (edge.attr1 != 0)
			{
				return true;
			}

			unsigned int i = vertices.size();
			vertices.push_back(vml_vec4(source.position.x, source.position.y, 0.0f, 1.0f));
			vertices.push_back(vml_vec4(destination.position.x, destination.position.y, 0.0f, 1.0f));
			colors.push_back(color);
			colors.push_back(color);
			indices.push_back(i);
			indices.push_back(i + 1);

			return true;
		}

	};

	unsigned int buffers[3];
	unsigned int vao;
	unsigned int elementsCount;
	bool built;

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

	void build()
	{
		if (!built)
		{
			glGenBuffers(3, buffers);
			glGenVertexArrays(1, &vao);
		}

		std::vector<vml_vec4> vertices;
		std::vector<vml_vec4> colors;
		std::vector<unsigned int> indices;
		for (unsigned int i = 0; i < g_numExtractedPrimitives; i++)
		{
			RoadNetworkGraph::Primitive& primitive = g_primitives[i];

			switch (primitive.type)
			{
			case RoadNetworkGraph::MINIMAL_CYCLE:
				for (unsigned int j = 0; j < primitive.numVertices; j++)
				{
					vml_vec2& v0 = primitive.vertices[j];
					vml_vec2 v1;
					if (j == primitive.numVertices - 1)
					{
						v1 = primitive.vertices[0];
					}
					else
					{
						v1 = primitive.vertices[j + 1];
					}

					unsigned int k = vertices.size();

					vertices.push_back(vml_vec4(v0.x, v0.y, 0.0f, 1.0f));
					vertices.push_back(vml_vec4(v1.x, v1.y, 0.0f, 1.0f));

					colors.push_back(g_configuration->cycleColor);
					colors.push_back(g_configuration->cycleColor);

					indices.push_back(k);
					indices.push_back(k + 1);
				}
				break;
			case RoadNetworkGraph::FILAMENT:
				for (unsigned int j = 0; j < primitive.numVertices - 1; j++)
				{
					vml_vec2& v0 = primitive.vertices[j];
					vml_vec2& v1 = primitive.vertices[j + 1];

					unsigned int k = vertices.size();

					vertices.push_back(vml_vec4(v0.x, v0.y, 0.0f, 1.0f));
					vertices.push_back(vml_vec4(v1.x, v1.y, 0.0f, 1.0f));

					colors.push_back(g_configuration->filamentColor);
					colors.push_back(g_configuration->filamentColor);

					indices.push_back(k);
					indices.push_back(k + 1);
				}
				break;
			case RoadNetworkGraph::ISOLATED_VERTEX:
				unsigned int k = vertices.size();
				vml_vec2& v0 = primitive.vertices[0];
				vertices.push_back(vml_vec4(v0.x, v0.y, 0.0f, 1.0f));
				colors.push_back(g_configuration->isolatedVertexColor);
				indices.push_back(k);
			}
		}

		RoadNetworkGraph::traverse(g_graph, GeometryCreationTraversal(vertices, colors, indices, g_configuration->streetColor));

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

};

#endif