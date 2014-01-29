#ifndef ROADNETWORKGEOMETRYGENERATOR_H
#define ROADNETWORKGEOMETRYGENERATOR_H

#include <Configuration.cuh>
#include <Geometry.h>
#include <RoadNetworkGraphGenerationObserver.h>
#include <GraphTraversal.h>
#include <vector_math.h>

#include <GL3/gl3w.h>

#include <exception>

class RoadNetworkGeometryGenerator : public Geometry, public RoadNetworkGraphGenerationObserver
{
private:
	struct GeometryCreationTraversal : public RoadNetworkGraph::GraphTraversal
	{
		vml_vec4* vertices;
		vml_vec4* colors;
		unsigned int* indices;
		vml_vec4 color;
		unsigned int lastVerticesIndex;
		unsigned int lastIndicesIndex;

		GeometryCreationTraversal(vml_vec4* vertices, vml_vec4* colors, unsigned int* indices, const vml_vec4& color, unsigned int lastVerticesIndex, unsigned int lastIndicesIndex) : 
			vertices(vertices),
			colors(colors),
			indices(indices),
			color(color),
			lastVerticesIndex(lastVerticesIndex), 
			lastIndicesIndex(lastIndicesIndex) 
		{
		}

		~GeometryCreationTraversal() 
		{
		}

		virtual bool operator () (const RoadNetworkGraph::Vertex& source, const RoadNetworkGraph::Vertex& destination, const RoadNetworkGraph::Edge& edge)
		{
			if (edge.attr1 != 0)
			{
				return true;
			}

			vertices[lastVerticesIndex] = vml_vec4(source.getPosition().x, source.getPosition().y, 0.0f, 1.0f);
			vertices[lastVerticesIndex + 1] = vml_vec4(destination.getPosition().x, destination.getPosition().y, 0.0f, 1.0f);

			colors[lastVerticesIndex] = color;
			colors[lastVerticesIndex + 1] = color;

			indices[lastIndicesIndex] = lastVerticesIndex;
			indices[lastIndicesIndex + 1] = lastVerticesIndex + 1;

			lastVerticesIndex += 2;
			lastIndicesIndex += 2;

			return true;
		}

	};

	unsigned int buffers[3];
	unsigned int vao;
	unsigned int elementsCount;
	vml_vec4* vertices;
	vml_vec4* colors;
	unsigned int* indices;
	unsigned int vertexBufferSize;
	unsigned int indexBufferSize;
	vml_vec4 cycleColor;
	vml_vec4 filamentColor;
	vml_vec4 isolatedVertexColor;
	vml_vec4 streetColor;
	bool built;

	void disposeBuffers() 
	{
		glDeleteBuffers(3, buffers);
		glDeleteVertexArrays(1, &vao);

		free(vertices);
		free(colors);
		free(indices);
	}

	void createBuffers() 
	{
		glGenBuffers(3, buffers);
		glGenVertexArrays(1, &vao);

		vertices = (vml_vec4*)malloc(sizeof(vml_vec4) * vertexBufferSize);
		colors = (vml_vec4*)malloc(sizeof(vml_vec4) * vertexBufferSize);
		indices = (unsigned int*)malloc(sizeof(unsigned int) * indexBufferSize);
	}

public:
	RoadNetworkGeometryGenerator() : 
		built(false), 
		vertices(0), 
		colors(0), 
		indices(0), 
		vertexBufferSize(0),
		indexBufferSize(0),
		elementsCount(0)
	{
	}

	~RoadNetworkGeometryGenerator()
	{
		if (built)
		{
			disposeBuffers();
		}
	}

	void readConfigurations(const Configuration& configuration)
	{
		if (configuration.vertexBufferSize != vertexBufferSize ||
			configuration.indexBufferSize != indexBufferSize)
		{
			vertexBufferSize = configuration.vertexBufferSize;
			indexBufferSize = configuration.indexBufferSize;

			if (built)
			{
				disposeBuffers();
			}
			createBuffers();
		}

		cycleColor = configuration.cycleColor;
		filamentColor = configuration.filamentColor;
		isolatedVertexColor = configuration.isolatedVertexColor;
		streetColor = configuration.streetColor;
	}

	virtual void update(RoadNetworkGraph::Graph* graph, unsigned int numPrimitives, RoadNetworkGraph::Primitive* primitives)
	{
		// FIXME: checking invariants
		if (!built)
		{
			throw std::exception("!built");
		}

		unsigned int lastVerticesIndex = 0, lastIndicesIndex = 0;
		for (unsigned int j = 0; j < numPrimitives; j++)
		{
			RoadNetworkGraph::Primitive& primitive = primitives[j];

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

					vertices[lastVerticesIndex] = vml_vec4(v0.x, v0.y, 0.0f, 1.0f);
					vertices[lastVerticesIndex + 1] = vml_vec4(v1.x, v1.y, 0.0f, 1.0f);

					colors[lastVerticesIndex] = cycleColor;
					colors[lastVerticesIndex + 1] = cycleColor;

					indices[lastIndicesIndex] = lastVerticesIndex;
					indices[lastIndicesIndex + 1] = lastVerticesIndex + 1;

					lastVerticesIndex += 2;
					lastIndicesIndex += 2;
				}
				break;
			case RoadNetworkGraph::FILAMENT:
				for (unsigned int k = 0; k < primitive.numVertices - 1; k++)
				{
					vml_vec2& v0 = primitive.vertices[k];
					vml_vec2& v1 = primitive.vertices[k + 1];

					vertices[lastVerticesIndex] = vml_vec4(v0.x, v0.y, 0.0f, 1.0f);
					vertices[lastVerticesIndex + 1] = vml_vec4(v1.x, v1.y, 0.0f, 1.0f);

					colors[lastVerticesIndex] = filamentColor;
					colors[lastVerticesIndex + 1] = filamentColor;

					indices[lastIndicesIndex] = lastVerticesIndex;
					indices[lastIndicesIndex + 1] = lastVerticesIndex + 1;

					lastVerticesIndex += 2;
					lastIndicesIndex += 2;
				}
				break;
			case RoadNetworkGraph::ISOLATED_VERTEX:
				vml_vec2& v0 = primitive.vertices[0];
				vertices[lastVerticesIndex] = vml_vec4(v0.x, v0.y, 0.0f, 1.0f);
				colors[lastVerticesIndex] = isolatedVertexColor;
				indices[lastIndicesIndex++] = lastVerticesIndex++;
				break;
			}
		}

		GeometryCreationTraversal traversal(vertices, colors, indices, streetColor, lastVerticesIndex, lastIndicesIndex);
		RoadNetworkGraph::traverse(graph, traversal);

		glBindBuffer(GL_ARRAY_BUFFER, buffers[0]);
		glBufferData(GL_ARRAY_BUFFER, traversal.lastVerticesIndex * sizeof(vml_vec4), (void*)vertices, GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glBindBuffer(GL_ARRAY_BUFFER, buffers[1]);
		glBufferData(GL_ARRAY_BUFFER, traversal.lastVerticesIndex * sizeof(vml_vec4), (void*)colors, GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		elementsCount = traversal.lastIndicesIndex;

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffers[2]);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, elementsCount * sizeof(unsigned int), (void*)indices, GL_STATIC_DRAW);
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