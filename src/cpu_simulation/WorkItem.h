#ifndef WORKITEM_H
#define WORKITEM_H

template<typename T>
struct WorkItem
{
	int operationCode;
	T data;

	WorkItem(int operationCode, T& data) : operationCode(operationCode), data(data) {}

};

#endif