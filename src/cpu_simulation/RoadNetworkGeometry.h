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
		unsigned int lastVerticesIndex;
		unsigned int lastIndicesIndex;

		GeometryCreationTraversal(unsigned int lastVerticesIndex, unsigned int lastIndicesIndex) : lastVerticesIndex(lastVerticesIndex), lastIndicesIndex(lastIndicesIndex) {}
		~GeometryCreationTraversal() {}

		virtual bool operator () (const RoadNetworkGraph::Vertex& source, const RoadNetworkGraph::Vertex& destination, const RoadNetworkGraph::Edge& edge)
		{
			if (edge.attr1 != 0)
			{
				return true;
			}

			g_verticesBuffer[lastVerticesIndex] = vml_vec4(source.position.x, source.position.y, 0.0f, 1.0f);
			g_verticesBuffer[lastVerticesIndex + 1] = vml_vec4(destination.position.x, destination.position.y, 0.0f, 1.0f);

			g_colorsBuffer[lastVerticesIndex] = g_configuration->streetColor;
			g_colorsBuffer[lastVerticesIndex + 1] = g_configuration->streetColor;

			g_indicesBuffer[lastIndicesIndex] = lastVerticesIndex;
			g_indicesBuffer[lastIndicesIndex + 1] = lastVerticesIndex + 1;

			lastVerticesIndex += 2;
			lastIndicesIndex += 2;

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

		unsigned int lastVerticesIndex = 0, lastIndicesIndex = 0;
		for (unsigned int j = 0; j < g_numExtractedPrimitives; j++)
		{
			RoadNetworkGraph::Primitive& primitive = g_primitives[j];

			switch (primitive.type)
			{
			case RoadNetworkGraph::MINIMAL_CYCLE:
				for (unsigned int k = 0; k < primitive.numVertices; k++)
				{
					vml_vec2& v0 = primitive.vertices[k];
					vml_vec2 v1;
					if (k == primitive.numVertices - 1)
					{
						v1 = primitive.vertices[0];
					}
					else
					{
						v1 = primitive.vertices[k + 1];
					}

					g_verticesBuffer[lastVerticesIndex] = vml_vec4(v0.x, v0.y, 0.0f, 1.0f);
					g_verticesBuffer[lastVerticesIndex + 1] = vml_vec4(v1.x, v1.y, 0.0f, 1.0f);

					g_colorsBuffer[lastVerticesIndex] = g_configuration->cycleColor;
					g_colorsBuffer[lastVerticesIndex + 1] = g_configuration->cycleColor;

					g_indicesBuffer[lastIndicesIndex] = lastVerticesIndex;
					g_indicesBuffer[lastIndicesIndex + 1] = lastVerticesIndex + 1;

					lastVerticesIndex += 2;
					lastIndicesIndex += 2;
				}
				break;
			case RoadNetworkGraph::FILAMENT:
				for (unsigned int k = 0; k < primitive.numVertices - 1; k++)
				{
					vml_vec2& v0 = primitive.vertices[k];
					vml_vec2& v1 = primitive.vertices[k + 1];

					g_verticesBuffer[lastVerticesIndex] = vml_vec4(v0.x, v0.y, 0.0f, 1.0f);
					g_verticesBuffer[lastVerticesIndex + 1] = vml_vec4(v1.x, v1.y, 0.0f, 1.0f);

					g_colorsBuffer[lastVerticesIndex] = g_configuration->filamentColor;
					g_colorsBuffer[lastVerticesIndex + 1] = g_configuration->filamentColor;

					g_indicesBuffer[lastIndicesIndex] = lastVerticesIndex;
					g_indicesBuffer[lastIndicesIndex + 1] = lastVerticesIndex + 1;

					lastVerticesIndex += 2;
					lastIndicesIndex += 2;
				}
				break;
			case RoadNetworkGraph::ISOLATED_VERTEX:
				vml_vec2& v0 = primitive.vertices[0];
				g_verticesBuffer[lastVerticesIndex] = vml_vec4(v0.x, v0.y, 0.0f, 1.0f);
				g_colorsBuffer[lastVerticesIndex] = g_configuration->isolatedVertexColor;
				g_indicesBuffer[lastIndicesIndex++] = lastVerticesIndex++;
				break;
			}
		}

		GeometryCreationTraversal traversal(lastVerticesIndex, lastIndicesIndex);
		RoadNetworkGraph::traverse(g_graph, traversal);

		glBindBuffer(GL_ARRAY_BUFFER, buffers[0]);
		glBufferData(GL_ARRAY_BUFFER, traversal.lastVerticesIndex * sizeof(vml_vec4), (void*)g_verticesBuffer, GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glBindBuffer(GL_ARRAY_BUFFER, buffers[1]);
		glBufferData(GL_ARRAY_BUFFER, traversal.lastVerticesIndex * sizeof(vml_vec4), (void*)g_colorsBuffer, GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		elementsCount = traversal.lastIndicesIndex;

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffers[2]);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, elementsCount * sizeof(unsigned int), (void*)g_indicesBuffer, GL_STATIC_DRAW);
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