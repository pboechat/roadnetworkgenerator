#ifndef ROADNETWORKLABELSGENERATOR_H
#define ROADNETWORKLABELSGENERATOR_H

#include "Defines.h"
#include <RoadNetworkGraphGenerationObserver.h>
#include <GraphTraversal.h>
#include <vector_math.h>

#include <glFont.h>
#include <GL3/gl3w.h>

#include <vector>
#include <sstream>
#include <string>

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

	struct LabelGenerationTraversal : public RoadNetworkGraph::GraphTraversal
	{
		LabelGenerationTraversal(std::vector<Label>& labels) : labels(labels) {}
		~LabelGenerationTraversal() {}

		virtual bool operator () (const RoadNetworkGraph::Vertex& source, const RoadNetworkGraph::Vertex& destination, const RoadNetworkGraph::Edge& edge)
		{
			std::stringstream label;
			label << source.index;
			labels.push_back(Label(label.str(), source.getPosition(), VERTEX_LABEL_COLOR));
			label.str(std::string());
			label.clear();
			label << destination.index;
			labels.push_back(Label(label.str(), destination.getPosition(), VERTEX_LABEL_COLOR));
			label.str(std::string());
			label.clear();
			label << edge.index;
			vml_vec2 edgePosition = vml_mix(source.getPosition(), destination.getPosition(), 0.5f);
			labels.push_back(Label(label.str(), edgePosition, EDGE_LABEL_COLOR));
			return true;
		}

	private:
		std::vector<Label>& labels;

	};

	std::vector<Label> labels;
	GLFont font;
	unsigned int textureId;
	bool built;

public:
	RoadNetworkLabelsGenerator() : built(false), textureId(0)
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

	virtual void update(RoadNetworkGraph::Graph* graph, unsigned int numPrimitives, RoadNetworkGraph::Primitive* primitives)
	{
		if (!built)
		{
			glGenTextures(1, &textureId);
			font.create(FONT_FILE_PATH, textureId);
		}
		else 
		{
			labels.clear();
		}
		
		RoadNetworkGraph::traverse(graph, LabelGenerationTraversal(labels));
		built = true;
	}

	void draw(GLFont::FontSetter& setter)
	{
		if (!built)
		{
			return;
		}

		for (unsigned int i = 0; i < labels.size(); i++)
		{
			Label& label = labels[i];
			font.draw(label.value, label.position.x, label.position.y, 0.0f, 2.0f, &label.color[0], setter);
		}
	}

};

#endif