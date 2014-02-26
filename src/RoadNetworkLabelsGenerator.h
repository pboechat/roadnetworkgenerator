#ifndef ROADNETWORKLABELSGENERATOR_H
#define ROADNETWORKLABELSGENERATOR_H

#pragma once

#include <Constants.h>
#include <Configuration.h>
#include <RoadNetworkGraphGenerationObserver.h>
#include <GraphTraversal.h>
#include <VectorMath.h>

#include <glFont.h>
#include <GL3/gl3w.h>

#include <vector>
#include <algorithm>
#include <sstream>
#include <string>

#define clearStringStream(__stringStream) \
	__stringStream.str(std::string()); \
	__stringStream.clear()

#define drawLabels(__collection) \
	for (unsigned int i = 0; i < __collection.size(); i++) \
	{ \
		Label& label = __collection[i]; \
		font.draw(label.value, label.position.x, label.position.y, 0.0f, labelFontSize, &label.color[0], setter); \
	}

class RoadNetworkLabelsGenerator : public RoadNetworkGraphGenerationObserver
{
private:
	struct Label
	{
		std::string value;
		vml_vec2 position;
		vml_vec4 color;

		Label(const std::string& value, const vml_vec2& position, const vml_vec4& color) : value(value), position(position), color(color) {}
		
	};

	struct LabelGenerationTraversal : public GraphTraversal
	{
		LabelGenerationTraversal(const std::vector<vml_vec2>& spawnPoints, std::vector<Label>& spawnPointLabels, std::vector<Label>& graphLabels) : 
			spawnPoints(spawnPoints),
			spawnPointLabels(spawnPointLabels), 
			graphLabels(graphLabels) 
		{
		}
		~LabelGenerationTraversal() {}

		virtual bool operator () (const Vertex& source, const Vertex& destination, const Edge& edge)
		{
			std::stringstream labelValue;

			std::vector<vml_vec2>::const_iterator it = std::find(spawnPoints.begin(), spawnPoints.end(), source.getPosition());
			if (it != spawnPoints.end())
			{
				unsigned int index = std::distance(spawnPoints.begin(), it);
				labelValue << index;
				spawnPointLabels.push_back(Label(labelValue.str(), source.getPosition(), SPAWN_POINT_LABEL_COLOR));
				clearStringStream(labelValue);
			}

			labelValue << source.index;
			graphLabels.push_back(Label(labelValue.str(), source.getPosition(), VERTEX_LABEL_COLOR));
			clearStringStream(labelValue);

			labelValue << destination.index;
			graphLabels.push_back(Label(labelValue.str(), destination.getPosition(), VERTEX_LABEL_COLOR));
			clearStringStream(labelValue);

			labelValue << edge.index;
			vml_vec2 edgePosition = vml_mix(source.getPosition(), destination.getPosition(), 0.5f);
			graphLabels.push_back(Label(labelValue.str(), edgePosition, EDGE_LABEL_COLOR));

			return true;
		}

	private:
		const std::vector<vml_vec2>& spawnPoints;
		std::vector<Label>& spawnPointLabels;
		std::vector<Label>& graphLabels;

	};

	std::vector<Label> spawnPointLabels;
	std::vector<Label> graphLabels;
	std::vector<vml_vec2> spawnPoints;
	GLFont font;
	unsigned int textureId;
	bool built;
	float labelFontSize;
	bool drawSpawnPointLabels;
	bool drawGraphLabels;

public:
	RoadNetworkLabelsGenerator() : 
		textureId(0),
		built(false), 
		labelFontSize(2.0f),
		drawSpawnPointLabels(false),
		drawGraphLabels(false)
	{
	}
	
	~RoadNetworkLabelsGenerator()
	{
		if (textureId != 0)
		{
			glDeleteTextures(1, &textureId);
			textureId = 0;
		}
	}

	void readConfigurations(const Configuration& configuration)
	{
		drawSpawnPointLabels = configuration.drawSpawnPointLabels;
		drawGraphLabels = configuration.drawGraphLabels;
		labelFontSize = configuration.labelFontSize;
		spawnPoints.clear();
		spawnPoints.insert(spawnPoints.end(), configuration.spawnPoints, (configuration.spawnPoints + configuration.numSpawnPoints));
	}

	virtual void update(Graph* graph, unsigned int numPrimitives, Primitive* primitives)
	{
		if (!built)
		{
			glGenTextures(1, &textureId);
			font.create(FONT_FILE_PATH, textureId);
		}
		else 
		{
			spawnPointLabels.clear();
			graphLabels.clear();
		}
		
		traverse(graph, LabelGenerationTraversal(spawnPoints, spawnPointLabels, graphLabels));
		built = true;
	}

	void draw(GLFont::FontSetter& setter)
	{
		if (!built)
		{
			return;
		}

		if (drawSpawnPointLabels)
		{
			drawLabels(spawnPointLabels);
		}

		if (drawGraphLabels)
		{
			drawLabels(graphLabels);
		}
	}

};

#endif