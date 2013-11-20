#ifndef ROADNETWORKGENERATOR_H
#define ROADNETWORKGENERATOR_H

#include <Configuration.h>
#include <RoadNetworkGeometry.h>
#include <Procedure.h>
#include <WorkQueuesManager.h>
#include <EvaluateRoad.h>
#include <RoadAttributes.h>
#include <RuleAttributes.h>
#include <Road.h>
#include <QuadTree.h>

#include <glm/glm.hpp>

#ifdef _DEBUG
#include <iostream>
#endif

class RoadNetworkGenerator
{
public:
	RoadNetworkGenerator() {}
	~RoadNetworkGenerator() {}

	void execute(const Configuration& configuration, RoadNetworkGeometry& geometry)
	{
		AABB worldBounds(0, 0, (float)configuration.worldWidth, (float)configuration.worldHeight);
		quadtree = new QuadTree(worldBounds, (float)configuration.quadtreeCellArea);
		WorkQueuesManager<Procedure>* frontBuffer = &buffer1;
		WorkQueuesManager<Procedure>* backBuffer = &buffer2;
		// starting at the center of the world
		RoadAttributes initialRoadAttributes(glm::vec3(configuration.worldWidth / 2.0f, configuration.worldWidth / 2.0f, 0), configuration.highwayLength, configuration.highwayWidth, 0, true);
		RuleAttributes initialRuleAttributes;
		initialRuleAttributes.highwayBranchingDistance = configuration.minHighwayBranchingDistance;
		frontBuffer->addWorkItem(new EvaluateRoad(Road(0, initialRoadAttributes, initialRuleAttributes, UNASSIGNED)));
		int derivation = 0;

		while (frontBuffer->notEmpty() && derivation++ < configuration.maxDerivations)
		{
			frontBuffer->resetCursors();

			do
			{
				Procedure* procedure;

				while ((procedure = frontBuffer->popWorkItem()) != 0)
				{
					procedure->execute(*backBuffer, *quadtree, configuration);
					delete procedure;
				}
			}
			while (frontBuffer->nextWorkQueue());

			std::swap(frontBuffer, backBuffer);
#ifdef _DEBUG
			std::cout << "derivation " << derivation << std::endl;
#endif
		}

		frontBuffer->clear();
		backBuffer->clear();
		std::vector<Line> lines;
		quadtree->query(worldBounds, lines);
#ifdef _DEBUG
		std::cout << "no. lines: " << lines.size() << std::endl;
#endif
#ifdef _DRAW_QUADTREE
		geometry.build(configuration, lines, *quadtree);
#else
		geometry.build(configuration, lines);
#endif
		delete quadtree;
	}

private:
	WorkQueuesManager<Procedure> buffer1;
	WorkQueuesManager<Procedure> buffer2;
	QuadTree* quadtree;

};

#endif