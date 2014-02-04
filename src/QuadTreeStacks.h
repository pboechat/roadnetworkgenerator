#ifndef QUADTREESTACKS_H
#define QUADTREESTACKS_H

#include <Constants.h>
#include <CpuGpuCompatibility.h>
#include <Box2D.h>

struct SimpleQuadTreeStack
{
private:
	struct Entry
	{
		unsigned int index;
		unsigned int offset;
		unsigned int levelWidth;

	};

	unsigned int count;
	Entry data[QUADTREE_STACK_DATA_SIZE];

public:
	HOST_AND_DEVICE_CODE void push(unsigned int index, unsigned int offset, unsigned int levelWidth)
	{
		if (count > QUADTREE_STACK_DATA_SIZE)
		{
			THROW_EXCEPTION("QuadTreeStack: count > QUADTREE_STACK_DATA_SIZE");
		}

		Entry& entry = data[count++];
		entry.index = index;
		entry.offset = offset;
		entry.levelWidth = levelWidth;
	}

	HOST_AND_DEVICE_CODE void pop(unsigned int& index, unsigned int& offset, unsigned int& levelWidth)
	{
		if (count < 1)
		{
			THROW_EXCEPTION("QuadTreeStack: count < 1");
		}

		Entry& entry = data[--count];
		index = entry.index;
		offset = entry.offset;
		levelWidth = entry.levelWidth;
	}

	inline HOST_AND_DEVICE_CODE bool notEmpty() const
	{
		return count > 0;
	}

	HOST_AND_DEVICE_CODE SimpleQuadTreeStack() : count(0) {}
	HOST_AND_DEVICE_CODE ~SimpleQuadTreeStack() {}

};

struct InitializationQuadTreeStack
{
private:
	struct Entry
	{
		unsigned int index;
		unsigned int offset;
		unsigned int levelWidth;
		unsigned int depth;
		Box2D quadrantBounds;

	};

	unsigned int count;
	Entry data[QUADTREE_STACK_DATA_SIZE];
	
public:
	HOST_AND_DEVICE_CODE void push(unsigned int index, unsigned int offset, unsigned int levelWidth, unsigned int depth, const Box2D& quadrantBounds)
	{
		if (count > QUADTREE_STACK_DATA_SIZE)
		{
			THROW_EXCEPTION("QuadTreeStack: count > QUADTREE_STACK_DATA_SIZE");
		}

		Entry& entry = data[count++];
		entry.index = index;
		entry.offset = offset;
		entry.levelWidth = levelWidth;
		entry.depth = depth;
		entry.quadrantBounds = quadrantBounds;
	}

	HOST_AND_DEVICE_CODE void pop(unsigned int& index, unsigned int& offset, unsigned int& levelWidth, unsigned int& depth, Box2D& quadrantBounds)
	{
		if (count < 1)
		{
			THROW_EXCEPTION("QuadTreeStack: count < 1");
		}

		Entry& entry = data[--count];
		index = entry.index;
		offset = entry.offset;
		levelWidth = entry.levelWidth;
		depth = entry.depth;
		quadrantBounds = entry.quadrantBounds;
	}

	inline HOST_AND_DEVICE_CODE bool notEmpty() const
	{
		return count > 0;
	}

	HOST_AND_DEVICE_CODE InitializationQuadTreeStack() : count(0) {}
	HOST_AND_DEVICE_CODE ~InitializationQuadTreeStack() {}

};

#endif