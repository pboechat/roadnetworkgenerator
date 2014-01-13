#ifndef ROADNETWORKGENERATOR_H
#define ROADNETWORKGENERATOR_H

#include "Defines.h"
#include <RoadNetworkGeometry.h>
#include <Globals.h>
#include <ProceduresCodes.h>
#include <WorkQueuesSet.h>
#include <Road.h>
#include <RoadAttributes.h>
#include <StreetRuleAttributes.h>
#include <HighwayRuleAttributes.h>
#include <MathExtras.h>
#include <MinimalCycleBasis.h>

#include <vector_math.h>

class RoadNetworkGenerator
{
public:
	RoadNetworkGenerator() : buffer1(g_workQueues1, NUM_PROCEDURES), buffer2(g_workQueues2, NUM_PROCEDURES), lastHighwayDerivation(0), lastStreetDerivation(0)
#ifdef _DEBUG
		, maxWorkQueueCapacityUsed(0)
#endif
	{}
	~RoadNetworkGenerator() {}

	void execute()
	{
		WorkQueuesSet* frontBuffer = &buffer1;
		WorkQueuesSet* backBuffer = &buffer2;

		// set highway spawn points
		for (unsigned int i = 0; i < g_configuration->numSpawnPoints; i++)
		{
			vml_vec2 spawnPoint = g_configuration->spawnPoints[i];
			RoadNetworkGraph::VertexIndex source = RoadNetworkGraph::createVertex(g_graph, spawnPoint);
			frontBuffer->addWorkItem(EVALUATE_HIGHWAY, Road<HighwayRuleAttributes>(0, RoadAttributes(source, g_configuration->highwayLength, 0, true), HighwayRuleAttributes(), UNASSIGNED));
			frontBuffer->addWorkItem(EVALUATE_HIGHWAY, Road<HighwayRuleAttributes>(0, RoadAttributes(source, g_configuration->highwayLength, -MathExtras::HALF_PI, true), HighwayRuleAttributes(), UNASSIGNED));
			frontBuffer->addWorkItem(EVALUATE_HIGHWAY, Road<HighwayRuleAttributes>(0, RoadAttributes(source, g_configuration->highwayLength, MathExtras::HALF_PI, true), HighwayRuleAttributes(), UNASSIGNED));
			frontBuffer->addWorkItem(EVALUATE_HIGHWAY, Road<HighwayRuleAttributes>(0, RoadAttributes(source, g_configuration->highwayLength, MathExtras::PI, true), HighwayRuleAttributes(), UNASSIGNED));
		}

		lastHighwayDerivation = 0;
		while (frontBuffer->notEmpty() && lastHighwayDerivation++ < g_configuration->maxHighwayDerivation)
		{
#ifdef _DEBUG
			if (frontBuffer->getNumWorkItems() > maxWorkQueueCapacityUsed)
			{
				maxWorkQueueCapacityUsed = frontBuffer->getNumWorkItems();
			}
#endif
			frontBuffer->executeAllWorkItems(backBuffer);
			std::swap(frontBuffer, backBuffer);
		}

		// extract the allotments
		RoadNetworkGraph::allocateExtractionBuffers(g_configuration->maxVertices, g_configuration->maxEdgeSequences, g_configuration->maxVisitedVertices);
		g_numExtractedPrimitives = RoadNetworkGraph::extractPrimitives(g_graph, g_primitives, g_configuration->maxPrimitives);
		RoadNetworkGraph::freeExtractionBuffers();

		buffer1.clear();
		buffer2.clear();

		RoadNetworkGraph::clear(g_graph);
#ifdef USE_QUADTREE
		RoadNetworkGraph::clear(g_quadtree);
#endif

		frontBuffer = &buffer1;
		backBuffer = &buffer2;

		// set street spawn points
		for (unsigned int i = 0; i < g_numExtractedPrimitives; i++)
		{
			RoadNetworkGraph::Primitive& primitive = g_primitives[i];

			if (primitive.type != RoadNetworkGraph::MINIMAL_CYCLE)
			{
				continue;
			}

			vml_vec2 center = MathExtras::getBarycenter(primitive.vertices, primitive.numVertices);

			RoadNetworkGraph::VertexIndex source = RoadNetworkGraph::createVertex(g_graph, center);
			frontBuffer->addWorkItem(EVALUATE_STREET, Road<StreetRuleAttributes>(0, RoadAttributes(source, g_configuration->streetLength, -MathExtras::HALF_PI, false), StreetRuleAttributes(), UNASSIGNED));
			frontBuffer->addWorkItem(EVALUATE_STREET, Road<StreetRuleAttributes>(0, RoadAttributes(source, g_configuration->streetLength, MathExtras::HALF_PI, false), StreetRuleAttributes(), UNASSIGNED));
		}

		lastStreetDerivation = 0;
		while (frontBuffer->notEmpty() && lastStreetDerivation++ < g_configuration->maxStreetDerivation)
		{
#ifdef _DEBUG
			if (frontBuffer->getNumWorkItems() > maxWorkQueueCapacityUsed)
			{
				maxWorkQueueCapacityUsed = frontBuffer->getNumWorkItems();
			}
#endif
			frontBuffer->executeAllWorkItems(backBuffer);
			std::swap(frontBuffer, backBuffer);
		}

		buffer1.clear();
		buffer2.clear();
	}

#ifdef _DEBUG
	inline unsigned int getLastHighwayDerivation() const
	{
		return lastHighwayDerivation - 1;
	}

	inline unsigned int getLastStreetDerivation() const
	{
		return lastStreetDerivation - 1;
	}

	inline unsigned int getMaxWorkQueueCapacityUsed() const
	{
		return maxWorkQueueCapacityUsed;
	}
#endif

private:
	WorkQueuesSet buffer1;
	WorkQueuesSet buffer2;
	unsigned int maxWorkQueueCapacity;
	unsigned int lastHighwayDerivation;
	unsigned int lastStreetDerivation;
#ifdef _DEBUG
	unsigned int maxWorkQueueCapacityUsed;
#endif

};

#endif