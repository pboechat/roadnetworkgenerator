#ifndef WORKITEM_H
#define WORKITEM_H

struct WorkItem
{
	virtual unsigned int getCode() const
	{
		return 0;
	}

};

#endif