#ifndef ROADNETWORKGEOMETRYGENERATOR_H
#define ROADNETWORKGEOMETRYGENERATOR_H

#pragma once

#include <Configuration.h>
#include <Geometry.h>
#include <RoadNetworkGraphGenerationObserver.h>
#include <GraphTraversal.h>
#include <GraphTraversalFunctions.h>
#include <VectorMath.h>

#include <GL3/gl3w.h>

#include <exception>

class RoadNetworkGeometryGenerator : public RoadNetworkGraphGenerationObserver
{
private:
	struct GeometryCreationTraversal : public GraphTraversal
	{
		vml_vec4* vertices;
		vml_vec4* colors;
		unsigned int* indices;
		vml_vec4 color;
		unsigned int maxNumVertices;
		unsigned int maxNumIndices;
		unsigned int lastVerticesIndex;
		unsigned int lastElementIndex;

		GeometryCreationTraversal(vml_vec4* vertices, vml_vec4* colors, unsigned int* indices, const vml_vec4& color, unsigned int maxNumVertices, unsigned int maxNumIndices, unsigned int lastVerticesIndex, unsigned int lastElementIndex) : 
			vertices(vertices),
			colors(colors),
			indices(indices),
			color(color),
			maxNumVertices(maxNumVertices),
			maxNumIndices(maxNumIndices),
			lastVerticesIndex(lastVerticesIndex), 
			lastElementIndex(lastElementIndex) 
		{
		}

		~GeometryCreationTraversal() 
		{
		}

		virtual bool operator () (const Vertex& source, const Vertex& destination, const Edge& edge)
		{
			// secondary roadnetwork edges only
			if (edge.attr1 != 0)
			{
				return true;
			}

			// FIXME: checking boundaries
			if (lastVerticesIndex + 1 >= maxNumVertices)
			{
				throw std::exception("max. number of vertices in vertex buffer overflow");
			}

			vertices[lastVerticesIndex] = vml_vec4(source.getPosition().x, source.getPosition().y, 0.0f, 1.0f);
			vertices[lastVerticesIndex + 1] = vml_vec4(destination.getPosition().x, destination.getPosition().y, 0.0f, 1.0f);

			colors[lastVerticesIndex] = color;
			colors[lastVerticesIndex + 1] = color;

			// FIXME: checking boundaries
			if (lastElementIndex + 1 >= maxNumIndices)
			{
				throw std::exception("max. number of indices in index buffer overflow");
			}

			indices[lastElementIndex] = lastVerticesIndex;
			indices[lastElementIndex + 1] = lastVerticesIndex + 1;

			lastVerticesIndex += 2;
			lastElementIndex += 2;

			return true;
		}

	};

	unsigned int buffers[3];
	unsigned int vao;
	unsigned int firstQuadtreeEdgeIndex;
	unsigned int firstQuadtreeQuadsIndex;
	unsigned int lastElementIndex;
	vml_vec4* vertices;
	vml_vec4* colors;
	unsigned int* indices;
	unsigned int vertexBufferSize;
	unsigned int indexBufferSize;
	vml_vec4 cycleColor;
	vml_vec4 filamentColor;
	vml_vec4 isolatedVertexColor;
	vml_vec4 streetColor;
	vml_vec4 quadtreeColor;
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
		firstQuadtreeEdgeIndex(0),
		firstQuadtreeQuadsIndex(0),
		lastElementIndex(0)
	{
	}

	~RoadNetworkGeometryGenerator()
	{
		if (built)
		{
			disposeBuffers();
			built = false;
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
			built = true;
		}

		cycleColor = configuration.getCycleColor();
		filamentColor = configuration.getFilamentColor();
		isolatedVertexColor = configuration.getIsolatedVertexColor();
		streetColor = configuration.getStreetColor();
		quadtreeColor = configuration.getQuadtreeColor();
	}

	virtual void update(Graph* graph, unsigned int numPrimitives, Primitive* primitives)
	{
		// FIXME: checking invariants
		if (!built)
		{
			throw std::exception("!built");
		}

		unsigned int lastVerticesIndex = 0;
		lastElementIndex = 0;
		for (unsigned int i = 0; i < numPrimitives; i++)
		{
			Primitive& primitive = primitives[i];

			if (primitive.removed)
			{
				continue;
			}

			for (unsigned int j = 0; j < primitive.numEdges; j++)
			{
				Edge& edge = graph->edges[primitive.edges[j]];

				vml_vec2 v0 = graph->vertices[edge.source].getPosition();
				vml_vec2 v1 = graph->vertices[edge.destination].getPosition();

				// FIXME: checking boundaries
				if (lastVerticesIndex + 1 >= vertexBufferSize)
				{
					throw std::exception("max. number of vertices in vertex buffer overflow");
				}

				vertices[lastVerticesIndex] = vml_vec4(v0.x, v0.y, 0.0f, 1.0f);
				vertices[lastVerticesIndex + 1] = vml_vec4(v1.x, v1.y, 0.0f, 1.0f);

				vml_vec4 color = (primitive.type == MINIMAL_CYCLE) ? cycleColor : filamentColor;

				colors[lastVerticesIndex] = color;
				colors[lastVerticesIndex + 1] = color;

				// FIXME: checking boundaries
				if (lastElementIndex + 1 >= indexBufferSize)
				{
					throw std::exception("max. number of indices in index buffer overflow");
				}

				indices[lastElementIndex] = lastVerticesIndex;
				indices[lastElementIndex + 1] = lastVerticesIndex + 1;

				lastVerticesIndex += 2;
				lastElementIndex += 2;
			}
		}

		GeometryCreationTraversal traversal(vertices, colors, indices, streetColor, vertexBufferSize, indexBufferSize, lastVerticesIndex, lastElementIndex);
		traverse(graph, traversal);

		lastVerticesIndex = traversal.lastVerticesIndex;
		lastElementIndex = traversal.lastElementIndex;

		firstQuadtreeEdgeIndex = lastElementIndex;

		for (unsigned int i = 0; i < graph->quadtree->totalNumQuadrants; i++)
		{
			Quadrant& quadrant = graph->quadtree->quadrants[i];

			if (!quadrant.hasEdges)
			{
				continue;
			}

			// FIXME: checking boundaries
			if (lastVerticesIndex + 4 >= vertexBufferSize)
			{
				throw std::exception("max. number of vertices in vertex buffer overflow");
			}

			vml_vec2 min = quadrant.bounds.getMin();
			vml_vec2 max = quadrant.bounds.getMax();

			vertices[lastVerticesIndex]		= vml_vec4(max.x, max.y, 0.1f, 1.0f);
			vertices[lastVerticesIndex + 1] = vml_vec4(min.x, max.y, 0.1f, 1.0f);
			vertices[lastVerticesIndex + 2] = vml_vec4(min.x, min.y, 0.1f, 1.0f);
			vertices[lastVerticesIndex + 3] = vml_vec4(max.x, min.y, 0.1f, 1.0f);

			colors[lastVerticesIndex]		= vml_vec4(0.0f, 1.0f, 0.0f, 1.0f);
			colors[lastVerticesIndex + 1]	= vml_vec4(0.0f, 1.0f, 0.0f, 1.0f);
			colors[lastVerticesIndex + 2]	= vml_vec4(0.0f, 1.0f, 0.0f, 1.0f);
			colors[lastVerticesIndex + 3]	= vml_vec4(0.0f, 1.0f, 0.0f, 1.0f);

			// FIXME: checking boundaries
			if (lastElementIndex + 8 >= indexBufferSize)
			{
				throw std::exception("max. number of indices in index buffer overflow");
			}

			indices[lastElementIndex]	  = lastVerticesIndex;
			indices[lastElementIndex + 1] = lastVerticesIndex + 1;
			indices[lastElementIndex + 2] = lastVerticesIndex + 1;
			indices[lastElementIndex + 3] = lastVerticesIndex + 2;
			indices[lastElementIndex + 4] = lastVerticesIndex + 2;
			indices[lastElementIndex + 5] = lastVerticesIndex + 3;
			indices[lastElementIndex + 6] = lastVerticesIndex + 3;
			indices[lastElementIndex + 7] = lastVerticesIndex;

			lastVerticesIndex += 4;
			lastElementIndex += 8;
		}

		firstQuadtreeQuadsIndex = lastElementIndex;
		for (unsigned int i = 0; i < graph->quadtree->totalNumQuadrants; i++)
		{
			Quadrant& quadrant = graph->quadtree->quadrants[i];

			if (quadrant.edges == -1)
			{
				continue;
			}

			QuadrantEdges& quadrantEdges = graph->quadtree->quadrantsEdges[quadrant.edges];

			if (quadrantEdges.lastEdgeIndex == 0)
			{
				continue;
			}

			// FIXME: checking boundaries
			if (lastVerticesIndex + 4 >= vertexBufferSize)
			{
				throw std::exception("max. number of vertices in vertex buffer overflow");
			}

			vml_vec2 min = quadrant.bounds.getMin();
			vml_vec2 max = quadrant.bounds.getMax();

			vertices[lastVerticesIndex] =	  vml_vec4(max.x, max.y, 0.1f, 1.0f);
			vertices[lastVerticesIndex + 1] = vml_vec4(min.x, max.y, 0.1f, 1.0f);
			vertices[lastVerticesIndex + 2] = vml_vec4(min.x, min.y, 0.1f, 1.0f);
			vertices[lastVerticesIndex + 3] = vml_vec4(max.x, min.y, 0.1f, 1.0f);

			colors[lastVerticesIndex] =		vml_vec4(0.0f, 1.0f, 0.0f, 0.1f);
			colors[lastVerticesIndex + 1] = vml_vec4(0.0f, 1.0f, 0.0f, 0.1f);
			colors[lastVerticesIndex + 2] = vml_vec4(0.0f, 1.0f, 0.0f, 0.1f);
			colors[lastVerticesIndex + 3] = vml_vec4(0.0f, 1.0f, 0.0f, 0.1f);

			// FIXME: checking boundaries
			if (lastElementIndex + 4 >= indexBufferSize)
			{
				throw std::exception("max. number of indices in index buffer overflow");
			}

			indices[lastElementIndex] = lastVerticesIndex;
			indices[lastElementIndex + 1] = lastVerticesIndex + 1;
			indices[lastElementIndex + 2] = lastVerticesIndex + 2;
			indices[lastElementIndex + 3] = lastVerticesIndex + 3;

			lastVerticesIndex += 4;
			lastElementIndex += 4;
		}

		glBindBuffer(GL_ARRAY_BUFFER, buffers[0]);
		glBufferData(GL_ARRAY_BUFFER, lastVerticesIndex * sizeof(vml_vec4), (void*)vertices, GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glBindBuffer(GL_ARRAY_BUFFER, buffers[1]);
		glBufferData(GL_ARRAY_BUFFER, lastVerticesIndex * sizeof(vml_vec4), (void*)colors, GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffers[2]);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, lastElementIndex * sizeof(unsigned int), (void*)indices, GL_STATIC_DRAW);
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

	void draw(bool drawQuadtree)
	{
		if (!built)
		{
			return;
		}

		glBindVertexArray(vao);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffers[2]);

		if (drawQuadtree)
		{
			glDrawRangeElements(GL_LINES, 0, firstQuadtreeQuadsIndex - 1, firstQuadtreeQuadsIndex, GL_UNSIGNED_INT, 0);
			glDrawRangeElements(GL_POINTS, 0, firstQuadtreeQuadsIndex - 1, firstQuadtreeQuadsIndex, GL_UNSIGNED_INT, 0);

			glDepthMask(GL_FALSE);
			glEnable(GL_BLEND);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

			glDrawRangeElements(GL_QUADS, 0, firstQuadtreeQuadsIndex - 1, lastElementIndex - firstQuadtreeQuadsIndex, GL_UNSIGNED_INT, (char*)(firstQuadtreeQuadsIndex * sizeof(unsigned int)));

			glDisable(GL_BLEND);
			glDepthMask(GL_TRUE);
		}
		else
		{
			glDrawRangeElements(GL_LINES, 0, firstQuadtreeEdgeIndex - 1, firstQuadtreeEdgeIndex, GL_UNSIGNED_INT, 0);
			glDrawRangeElements(GL_POINTS, 0, firstQuadtreeEdgeIndex - 1, firstQuadtreeEdgeIndex, GL_UNSIGNED_INT, 0);
		}

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		glBindVertexArray(0);
	}

};

#endif