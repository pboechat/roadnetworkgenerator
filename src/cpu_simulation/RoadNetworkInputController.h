#ifndef ROADNETWORKINPUTCONTROLLER_H
#define ROADNETWORKINPUTCONTROLLER_H

#include <InputController.h>
#include <SceneRenderer.h>
#include <Configuration.h>
#include <Camera.h>
#include <Graph.h>
#include <RoadNetworkGenerator.h>
#include <RoadNetworkGeometry.h>
#include <AABB.h>
#include <Timer.h>

#include <string>

#ifdef _DEBUG
#include <iostream>
#define toKilobytes(a) (a / 1024)
#define toMegabytes(a) (a / 1048576)
#endif

class RoadNetworkInputController : public InputController
{
public:
	RoadNetworkInputController(Camera& camera, 
							   const std::string& configurationFile,
							   SceneRenderer& sceneRenderer,
							   RoadNetworkGeometry& geometry) 
		: 
		InputController(camera, 100.0f, 10.0f),
		configurationFile(configurationFile),
		sceneRenderer(sceneRenderer),
		geometry(geometry)
	{
	}

	virtual void update(double deltaTime)
	{
		if (getKey(VK_ESCAPE))
		{
			Application::instance->halt();
		}

		if (getKey(VK_LEFT) || getKey(65))
		{
			moveCameraLeft((float)deltaTime);
		}

		else if (getKey(VK_RIGHT) || getKey(68))
		{
			moveCameraRight((float)deltaTime);
		}

		if (getKey(VK_UP) || getKey(87))
		{
			moveCameraUp((float)deltaTime);
		}

		else if (getKey(VK_DOWN) || getKey(83))
		{
			moveCameraDown((float)deltaTime);
		}

		if (getKey(81) || getKey(33))
		{
			moveCameraForward((float)deltaTime);
		}

		else if (getKey(69) || getKey(34))
		{
			moveCameraBackward((float)deltaTime);
		}

		if (getKey(VK_F5))
		{
			// reload configuration
			Configuration configuration;
			configuration.loadFromFile(configurationFile);
			AABB worldBounds(0.0f, 0.0f, (float)configuration.worldWidth, (float)configuration.worldHeight);
#ifdef USE_QUADTREE
			RoadNetworkGraph::Graph graph(worldBounds, configuration.quadtreeDepth, configuration.snapRadius, configuration.maxVertices, configuration.maxEdges, configuration.maxResultsPerQuery);
#else
			RoadNetworkGraph::Graph graph(worldBounds, configuration.snapRadius, configuration.maxVertices, configuration.maxEdges, configuration.maxResultsPerQuery);
#endif
			// regenerate road network graph
			RoadNetworkGenerator generator;
#ifdef _DEBUG
			Timer timer;
			timer.start();
#endif
			generator.execute(configuration, graph);
#ifdef _DEBUG
			timer.end();
			std::cout << "*****************************" << std::endl;
			std::cout << "	STATISTICS:				   " << std::endl;
			std::cout << "*****************************" << std::endl;
			std::cout << "generation time: " << timer.elapsedTime() << " seconds" << std::endl;
			std::cout << "memory (allocated/in use): " << toMegabytes(graph.getAllocatedMemory()) << " MB / " << toMegabytes(graph.getMemoryInUse()) << " MB" << std::endl;
			std::cout << "vertices (allocated/in use): " << graph.getAllocatedVertices() << " / " << graph.getVerticesInUse() << std::endl;
			std::cout << "edges (allocated/in use): " << graph.getAllocatedEdges() << " / " << graph.getEdgesInUse() << std::endl;
			std::cout << "vertex in connections (max./max. in use): " << graph.getMaxVertexInConnections() << " / " << graph.getMaxVertexInConnectionsInUse() << std::endl;
			std::cout << "vertex out connections (max./max. in use): " << graph.getMaxVertexOutConnections() << " / " << graph.getMaxVertexOutConnectionsInUse() << std::endl;
			std::cout << "avg. vertex in connections in use: " << graph.getAverageVertexInConnectionsInUse() << std::endl;
			std::cout << "avg. vertex out connections in use: " << graph.getAverageVertexOutConnectionsInUse() << std::endl;
#ifdef USE_QUADTREE
			std::cout << "edges per quadrant (max./max. in use): " << graph.getMaxEdgesPerQuadrant() << " / " << graph.getMaxEdgesPerQuadrantInUse() << std::endl;
#endif
			std::cout << "num. collision checks: " << graph.getNumCollisionChecks() << std::endl;
			std::cout  << std::endl << std::endl;
#endif
			// rebuild road network geometry
			geometry.build(graph, configuration.highwayColor, configuration.streetColor);
			sceneRenderer.setWorldBounds(worldBounds);
			camera.centerOnTarget(worldBounds);
		}

#ifdef _DEBUG
		if (getLeftMouseButtonDown())
		{
			glm::vec2 mousePosition = getMousePosition();
			std::cout << "(" << mousePosition.x << ", " << (camera.getScreenHeight() - mousePosition.y) << ")" << std::endl;
		}
#endif
	}

private:
	std::string configurationFile;
	SceneRenderer& sceneRenderer;
	RoadNetworkGeometry& geometry;

};

#endif