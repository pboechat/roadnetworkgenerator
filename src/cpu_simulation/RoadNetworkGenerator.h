#ifndef ROADNETWORKGENERATOR_H
#define ROADNETWORKGENERATOR_H

#include <Procedure.h>
#include <WorkQueuesManager.h>
#include <EvaluateRoad.h>

#include <vector>

class RoadNetworkGenerator
{
public:
	void execute()
	{
		WorkQueuesManager<Procedure>* currentBuffer = &frontBuffer;
		currentBuffer->addWorkItem(new EvaluateRoad());

		while (currentBuffer->notEmpty())
		{
			currentBuffer->resetCursors();

			do
			{
				Procedure* procedure;
				while ((procedure = currentBuffer->popWorkItem()) != 0)
				{
					procedure->execute(*currentBuffer, segments);
					delete procedure;
				}
			}
			while (currentBuffer->nextWorkQueue());

			currentBuffer = swapBuffer(currentBuffer);
		}
	}

private:
	WorkQueuesManager<Procedure> frontBuffer;
	WorkQueuesManager<Procedure> backBuffer;
	std::vector<Segment> segments;
	WorkQueuesManager<Procedure>* swapBuffer(WorkQueuesManager<Procedure>* currentBuffer)
	{
		if (currentBuffer == &frontBuffer)
		{
			return &backBuffer;
		}

		else
		{
			return &frontBuffer;
		}
	}
};

#endif