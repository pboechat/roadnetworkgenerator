#ifndef ROADNETWORKGENERATOR_H
#define ROADNETWORKGENERATOR_H

#include "Defines.h"
#include <RoadNetworkGeometry.h>
#include <Globals.h>
#include <ProceduresCodes.h>
#include <WorkQueuesSet.h>
#include <Road.h>
#include <RoadAttributes.h>

#include <MathExtras.h>
#include <MinimalCycleBasis.h>
#include <Box2D.h>
#include <ConvexHull.h>
#include <OBB2D.h>

#include <vector_math.h>

class RoadNetworkGenerator
{
public:
	RoadNetworkGenerator() : buffer1(g_dWorkQueues1, NUM_PROCEDURES), buffer2(g_dWorkQueues2, NUM_PROCEDURES), lastHighwayDerivation(0), lastStreetDerivation(0), maxPrimitiveSize(0)
#ifdef _DEBUG
		, maxWorkQueueCapacityUsed(0)
#endif
	{}
	~RoadNetworkGenerator() {}

	// TODO:
	void execute()
	{
		WorkQueuesSet* frontBuffer = &buffer1;
		WorkQueuesSet* backBuffer = &buffer2;

		// set highway spawn points
		for (unsigned int i = 0; i < g_dConfiguration->numSpawnPoints; i++)
		{
			vml_vec2 spawnPoint = g_dConfiguration->spawnPoints[i];
			RoadNetworkGraph::VertexIndex source = RoadNetworkGraph::createVertex(g_dGraph, spawnPoint);
			frontBuffer->addWorkItem(EVALUATE_HIGHWAY, Highway(0, RoadAttributes(source, g_dConfiguration->highwayLength, 0), UNASSIGNED));
			frontBuffer->addWorkItem(EVALUATE_HIGHWAY, Highway(0, RoadAttributes(source, g_dConfiguration->highwayLength, -MathExtras::HALF_PI), UNASSIGNED));
			frontBuffer->addWorkItem(EVALUATE_HIGHWAY, Highway(0, RoadAttributes(source, g_dConfiguration->highwayLength, MathExtras::HALF_PI), UNASSIGNED));
			frontBuffer->addWorkItem(EVALUATE_HIGHWAY, Highway(0, RoadAttributes(source, g_dConfiguration->highwayLength, MathExtras::PI), UNASSIGNED));
		}

		// generate highways
		lastHighwayDerivation = 0;
		while (frontBuffer->notEmpty() && lastHighwayDerivation++ < g_dConfiguration->maxHighwayDerivation)
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

		// -----------------------------------------------------------------------

		//////////////////////////////////////////////////////////////////////////
		//	HOST CODE
		//////////////////////////////////////////////////////////////////////////

		// initialize lightweight graph copy
		copyGraphToHost(g_hConfiguration->maxVertices, g_hConfiguration->maxEdges);
		
		// extract the allotments from graph copy
		RoadNetworkGraph::allocateExtractionBuffers(g_hConfiguration->maxVertices, g_hConfiguration->maxEdgeSequences, g_hConfiguration->maxVisitedVertices);
		g_hNumExtractedPrimitives = RoadNetworkGraph::extractPrimitives(g_hGraphCopy, g_hPrimitives, g_hConfiguration->maxPrimitives);
		RoadNetworkGraph::freeExtractionBuffers();

		buffer1.clear();
		buffer2.clear();

		frontBuffer = &buffer1;
		backBuffer = &buffer2;

		maxPrimitiveSize = 0;
		// set street spawn points
		for (unsigned int i = 0; i < g_hNumExtractedPrimitives; i++)
		{
			const RoadNetworkGraph::Primitive& primitive = g_hPrimitives[i];

			if (primitive.type != RoadNetworkGraph::MINIMAL_CYCLE)
			{
				continue;
			}

			maxPrimitiveSize = MathExtras::max<unsigned int>(maxPrimitiveSize, primitive.numVertices);

			vml_vec2 centroid;
			float area;
			MathExtras::getPolygonInfo(primitive.vertices, primitive.numVertices, area, centroid);
			if (area < g_dConfiguration->minBlockArea)
			{
				continue;
			}

			float angle;
			ConvexHull convexHull(primitive.vertices, primitive.numVertices);
			OBB2D obb(convexHull.hullPoints, convexHull.numHullPoints);
			angle = vml_angle(obb.axis[1], vml_vec2(0.0f, 1.0f));
			
			RoadNetworkGraph::VertexIndex source = RoadNetworkGraph::createVertex(g_hGraph, centroid);
			frontBuffer->addWorkItem(EVALUATE_STREET, Street(0, RoadAttributes(source, g_hConfiguration->streetLength, angle), UNASSIGNED));
			frontBuffer->addWorkItem(EVALUATE_STREET, Street(0, RoadAttributes(source, g_hConfiguration->streetLength, -MathExtras::HALF_PI + angle), UNASSIGNED));
			frontBuffer->addWorkItem(EVALUATE_STREET, Street(0, RoadAttributes(source, g_hConfiguration->streetLength, MathExtras::HALF_PI + angle), UNASSIGNED));
			frontBuffer->addWorkItem(EVALUATE_STREET, Street(0, RoadAttributes(source, g_hConfiguration->streetLength, MathExtras::PI + angle), UNASSIGNED));
		}

		copyGraphToDevice(g_hConfiguration->maxVertices, g_hConfiguration->maxEdges);

		//////////////////////////////////////////////////////////////////////////
		//	HOST CODE
		//////////////////////////////////////////////////////////////////////////

		// -----------------------------------------------------------------------

		//////////////////////////////////////////////////////////////////////////
		//	DEVICE CODE
		//////////////////////////////////////////////////////////////////////////

		// generate streets
		lastStreetDerivation = 0;
		while (frontBuffer->notEmpty() && lastStreetDerivation++ < g_dConfiguration->maxStreetDerivation)
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

	inline unsigned int getMaxPrimitiveSize() const
	{
		return maxPrimitiveSize;
	}
#endif

private:
	WorkQueuesSet buffer1;
	WorkQueuesSet buffer2;
	unsigned int maxWorkQueueCapacity;
	unsigned int maxPrimitiveSize;
	unsigned int lastHighwayDerivation;
	unsigned int lastStreetDerivation;
#ifdef _DEBUG
	unsigned int maxWorkQueueCapacityUsed;
#endif

};

#endif