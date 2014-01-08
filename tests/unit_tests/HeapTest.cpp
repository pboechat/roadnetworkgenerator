#include <Heap.h>

#include <gtest/gtest.h>

#define BUFFER_SIZE 14

int compare(const int& i0, const int& i1)
{
	if (i0 > i1)
	{
		return 1;
	}
	else if (i0 == i1)
	{
		return 0;
	}
	else
	{
		return -1;
	}
}

void setUpHeap(Heap<int>& heap)
{
	heap.insert(17);
	heap.insert(80);
	heap.insert(15);
	heap.insert(40);
	heap.insert(25);
	heap.insert(30);
	heap.insert(5);
	heap.insert(45);
	heap.insert(35);
	heap.insert(100);
	heap.insert(77);
	heap.insert(50);
	heap.insert(55);
	heap.insert(75);
}

TEST(heap, popFront)
{
	int buffer[BUFFER_SIZE];
	Heap<int> heap(buffer, BUFFER_SIZE, compare);

	setUpHeap(heap);

	EXPECT_EQ(5, heap.popFirst());
	EXPECT_EQ(15, heap.popFirst());
	EXPECT_EQ(17, heap.popFirst());
	EXPECT_EQ(25, heap.popFirst());
	EXPECT_EQ(30, heap.popFirst());
	EXPECT_EQ(35, heap.popFirst());
	EXPECT_EQ(40, heap.popFirst());
	EXPECT_EQ(45, heap.popFirst());
	EXPECT_EQ(50, heap.popFirst());
	EXPECT_EQ(55, heap.popFirst());
	EXPECT_EQ(75, heap.popFirst());
	EXPECT_EQ(77, heap.popFirst());
	EXPECT_EQ(80, heap.popFirst());
	EXPECT_EQ(100, heap.popFirst());
	EXPECT_TRUE(heap.empty());
}

/*TEST(heap, remove)
{
	int buffer[BUFFER_SIZE];
	Heap<int> heap(buffer, BUFFER_SIZE, compare);
	setUpHeap(heap);


}*/