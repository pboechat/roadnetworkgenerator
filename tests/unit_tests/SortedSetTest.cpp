#include <SortedSet.h>

#include <gtest/gtest.h>

#define BUFFER_SIZE 10

class SortedSetTest : public ::testing::Test 
{
private:
	struct IntComparer : public SortedSet<int>::Comparer
	{
		virtual int operator()(const int& i0, const int& i1) const
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

	} comparer;

protected:
	SortedSetTest() : sortedSet(buffer, BUFFER_SIZE, comparer) {}
	~SortedSetTest() {}

	int buffer[BUFFER_SIZE];
	SortedSet<int> sortedSet;

	virtual void SetUp() {
		sortedSet.insert(10);
		sortedSet.insert(5);
		sortedSet.insert(7);
		sortedSet.insert(3);
		sortedSet.insert(6);
		sortedSet.insert(2);
		sortedSet.insert(8);
		sortedSet.insert(1);
		sortedSet.insert(9);
		sortedSet.insert(4);
	}

	virtual void TearDown() {
	}

};

TEST_F(SortedSetTest, insert)
{
	EXPECT_EQ(10, sortedSet.size());
	EXPECT_EQ(1, sortedSet[0]);
	EXPECT_EQ(2, sortedSet[1]);
	EXPECT_EQ(3, sortedSet[2]);
	EXPECT_EQ(4, sortedSet[3]);
	EXPECT_EQ(5, sortedSet[4]);
	EXPECT_EQ(6, sortedSet[5]);
	EXPECT_EQ(7, sortedSet[6]);
	EXPECT_EQ(8, sortedSet[7]);
	EXPECT_EQ(9, sortedSet[8]);
	EXPECT_EQ(10, sortedSet[9]);
}

TEST_F(SortedSetTest, remove)
{
	EXPECT_EQ(10, sortedSet.size());
	sortedSet.remove(1);
	EXPECT_EQ(2, sortedSet[0]);
	sortedSet.remove(2);
	EXPECT_EQ(3, sortedSet[0]);
	sortedSet.remove(3);
	EXPECT_EQ(4, sortedSet[0]);
	sortedSet.remove(4);
	EXPECT_EQ(5, sortedSet[0]);
	sortedSet.remove(5);
	EXPECT_EQ(6, sortedSet[0]);
	sortedSet.remove(6);
	EXPECT_EQ(7, sortedSet[0]);
	sortedSet.remove(7);
	EXPECT_EQ(8, sortedSet[0]);
	sortedSet.remove(8);
	EXPECT_EQ(9, sortedSet[0]);
	sortedSet.remove(9);
	EXPECT_EQ(10, sortedSet[0]);
}
