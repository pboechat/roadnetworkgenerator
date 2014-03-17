#include <FileReader.h>
#include <StringUtils.h>

#include <sqlite3.h>

#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <exception>

#define ftimestamp									0
#define fconfig_name								1
#define fexpansion_kernel_blocks					2
#define fexpansion_kernel_threads					3
#define fcollision_detection_kernel_blocks			4
#define fcollision_detection_kernel_threads			5
#define fmax_highway_derivations					6
#define fmax_street_derivations						7
#define fquadtree_depth								8
#define fprimary_roadnetwork_expansion_time			9
#define fcollisions_computation_time				10
#define fprimitives_extraction_time					11
#define fsecondary_roadnetwork_expansion_time		12
#define fmemory_copy_gpu_cpu_time					13
#define fmemory_copy_cpu_gpu_time					14
#define fnum_vertices								15
#define fnum_primary_roadnetwork_edges				16
#define fnum_secondary_roadnetwork_edges			17
#define fnum_collisions								18
#define fmemory_in_use								19

struct Record
{
	std::string		timestamp;
	std::string		config_name;
	unsigned int	expansion_kernel_blocks;
	unsigned int	expansion_kernel_threads;
	unsigned int	collision_detection_kernel_blocks;
	unsigned int	collision_detection_kernel_threads;
	unsigned int	max_highway_derivations;
	unsigned int	max_street_derivations;
	unsigned int	quadtree_depth;
	float			primary_roadnetwork_expansion_time;
	float			collisions_computation_time;
	float			primitives_extraction_time;
	float			secondary_roadnetwork_expansion_time;
	float			memory_copy_gpu_cpu_time;
	float			memory_copy_cpu_gpu_time;
	unsigned int	num_vertices;
	unsigned int	num_primary_roadnetwork_edges;
	unsigned int	num_secondary_roadnetwork_edges;
	unsigned long	num_collisions;
	unsigned int	memory_in_use;

	Record() :
		timestamp(""),
		config_name(""),
		expansion_kernel_blocks(0),
		expansion_kernel_threads(0),
		collision_detection_kernel_blocks(0),
		collision_detection_kernel_threads(0),
		max_highway_derivations(0),
		max_street_derivations(0),
		quadtree_depth(0),
		primary_roadnetwork_expansion_time(0),
		collisions_computation_time(0),
		primitives_extraction_time(0),
		secondary_roadnetwork_expansion_time(0),
		memory_copy_gpu_cpu_time(0),
		memory_copy_cpu_gpu_time(0),
		num_vertices(0),
		num_primary_roadnetwork_edges(0),
		num_secondary_roadnetwork_edges(0),
		num_collisions(0),
		memory_in_use(0)
	{
	}

};

struct Group
{
	unsigned int	repetitions;
	Record*			records;

	void allocate(unsigned int repetitions)
	{
		this->repetitions						= repetitions;
		this->records							= new Record[repetitions];
	}

	Record average()
	{
		Record avg;
		avg.timestamp = records[0].timestamp;
		avg.config_name = records[0].config_name;
		for (unsigned int i = 0; i < repetitions; i++)
		{
			avg.expansion_kernel_blocks					+= records[i].expansion_kernel_blocks				;
			avg.expansion_kernel_threads				+= records[i].expansion_kernel_threads				;
			avg.collision_detection_kernel_blocks		+= records[i].collision_detection_kernel_blocks		;
			avg.collision_detection_kernel_threads		+= records[i].collision_detection_kernel_threads	;
			avg.max_highway_derivations					+= records[i].max_highway_derivations				;
			avg.max_street_derivations					+= records[i].max_street_derivations				;
			avg.quadtree_depth							+= records[i].quadtree_depth						;
			avg.primary_roadnetwork_expansion_time		+= records[i].primary_roadnetwork_expansion_time	;
			avg.collisions_computation_time				+= records[i].collisions_computation_time			;
			avg.primitives_extraction_time				+= records[i].primitives_extraction_time			;
			avg.secondary_roadnetwork_expansion_time	+= records[i].secondary_roadnetwork_expansion_time	;
			avg.memory_copy_gpu_cpu_time				+= records[i].memory_copy_gpu_cpu_time				;
			avg.memory_copy_cpu_gpu_time				+= records[i].memory_copy_cpu_gpu_time				;
			avg.num_vertices							+= records[i].num_vertices							;
			avg.num_primary_roadnetwork_edges			+= records[i].num_primary_roadnetwork_edges			;
			avg.num_secondary_roadnetwork_edges			+= records[i].num_secondary_roadnetwork_edges		;
			avg.num_collisions							+= records[i].num_collisions						;
			avg.memory_in_use							+= records[i].memory_in_use							;
		}

		avg.expansion_kernel_blocks					/= repetitions;
		avg.expansion_kernel_threads				/= repetitions;
		avg.collision_detection_kernel_blocks		/= repetitions;
		avg.collision_detection_kernel_threads		/= repetitions;
		avg.max_highway_derivations					/= repetitions;
		avg.max_street_derivations					/= repetitions;
		avg.quadtree_depth							/= repetitions;
		avg.primary_roadnetwork_expansion_time		/= repetitions;
		avg.collisions_computation_time				/= repetitions;
		avg.primitives_extraction_time				/= repetitions;
		avg.secondary_roadnetwork_expansion_time	/= repetitions;
		avg.memory_copy_gpu_cpu_time				/= repetitions;
		avg.memory_copy_cpu_gpu_time				/= repetitions;
		avg.num_vertices							/= repetitions;
		avg.num_primary_roadnetwork_edges			/= repetitions;
		avg.num_secondary_roadnetwork_edges			/= repetitions;
		avg.num_collisions							/= repetitions;
		avg.memory_in_use							/= repetitions;

		return avg;
	}

	void deallocate()
	{
		delete[] records;
	}

};

float toFloat(const std::string& str)
{
	std::string newStr = str;
	StringUtils::replace(newStr, ",", ".");
	return (float)atof(newStr.c_str());
}

std::string toString(float flt)
{
	std::stringstream sstream;
	sstream << flt;
	std::string str = sstream.str();
	StringUtils::replace(str, ".", ",");
	return str;
}

std::string toSQLiteString(const std::string& str)
{
	return std::string("'") + str + "'";
}

#define toInt(x) atoi(x.c_str())
#define toLong(x) atol(x.c_str())

void checkedSQLiteOpenCall(int result, sqlite3* db)
{
	if (result)
	{
		sqlite3_close(db);
		std::stringstream sstream;
		sstream << "can't open database: " << sqlite3_errmsg(db);
		throw std::exception(sstream.str().c_str());
	}
}

void checkedSQLiteCall(int result, char* errorMsg)
{
	if (result != SQLITE_OK)
	{
		sqlite3_free(errorMsg);
	}
}

int sqliteCallback(void *notUsed, int argc, char **argv, char **azColName)
{
	for (int i = 0; i < argc; i++)
	{
		printf("%s = %s\n", azColName[i], argv[i] ? argv[i] : "NULL");
	}
	printf("\n");

	return 0;
}

void outputToDatabase(std::string &outputFile, unsigned int numGroups, Group* groups)
{
	sqlite3 *db = 0;
	char* errorMsg;

	checkedSQLiteOpenCall(sqlite3_open(outputFile.c_str(), &db), db);

	for (unsigned int i = 0; i < numGroups; i++)
	{
		Record avg = groups[i].average();

		std::stringstream sqlCommand;
		sqlCommand << "insert into statistics values ("
			<< toSQLiteString(avg.timestamp)						<< ","
			<< toSQLiteString(avg.config_name)						<< ","
			<< avg.expansion_kernel_blocks							<< ","
			<< avg.expansion_kernel_threads							<< ","
			<< avg.collision_detection_kernel_blocks				<< ","
			<< avg.collision_detection_kernel_threads				<< ","
			<< avg.max_highway_derivations							<< ","
			<< avg.max_street_derivations							<< ","
			<< avg.quadtree_depth									<< ","
			<< avg.primary_roadnetwork_expansion_time				<< ","
			<< avg.collisions_computation_time						<< ","
			<< avg.primitives_extraction_time						<< ","
			<< avg.secondary_roadnetwork_expansion_time				<< ","
			<< avg.memory_copy_gpu_cpu_time							<< ","
			<< avg.memory_copy_cpu_gpu_time							<< ","
			<< avg.num_vertices										<< ","
			<< avg.num_primary_roadnetwork_edges					<< ","
			<< avg.num_secondary_roadnetwork_edges					<< ","
			<< avg.num_collisions									<< ","
			<< avg.memory_in_use									<< ");";

		sqlite3_exec(db, sqlCommand.str().c_str(), sqliteCallback, 0, &errorMsg);
	}
	
	sqlite3_close(db);
}

void outputToFile(std::string &outputFile, unsigned int numGroups, Group* groups)
{
	std::ofstream out;
	out.open(outputFile.c_str(), std::ios::out);

	out << "timestamp;" 
		<< "config_name;" 
		<< "expansion_kernel_blocks;" 
		<< "expansion_kernel_threads;" 
		<< "collision_detection_kernel_blocks;" 
		<< "collision_detection_kernel_threads;" 
		<< "max_highway_derivations;"
		<< "max_street_derivations;"
		<< "quadtree_depth;"
		<< "primary_roadnetwork_expansion_time;" 
		<< "collisions_computation_time;" 
		<< "primitives_extraction_time;" 
		<< "secondary_roadnetwork_expansion_time;" 
		<< "memory_copy_gpu_cpu_time;" 
		<< "memory_copy_cpu_gpu_time;" 
		<< "num_vertices;" 
		<< "num_primary_roadnetwork_edges;" 
		<< "num_secondary_roadnetwork_edges;" 
		<< "num_collisions;" 
		<< "memory_in_use;" 
		<< std::endl;

	for (unsigned int i = 0; i < numGroups; i++)
	{
		Record avg = groups[i].average();
		out << avg.timestamp										<< ";"
			<< avg.config_name										<< ";"
			<< avg.expansion_kernel_blocks							<< ";"
			<< avg.expansion_kernel_threads							<< ";"
			<< avg.collision_detection_kernel_blocks				<< ";"
			<< avg.collision_detection_kernel_threads				<< ";"
			<< avg.max_highway_derivations							<< ","
			<< avg.max_street_derivations							<< ","
			<< avg.quadtree_depth									<< ","
			<< toString(avg.primary_roadnetwork_expansion_time)		<< ";"
			<< toString(avg.collisions_computation_time)			<< ";"
			<< toString(avg.primitives_extraction_time)				<< ";"
			<< toString(avg.secondary_roadnetwork_expansion_time)	<< ";"
			<< toString(avg.memory_copy_gpu_cpu_time)				<< ";"
			<< toString(avg.memory_copy_cpu_gpu_time)				<< ";"
			<< avg.num_vertices										<< ";"
			<< avg.num_primary_roadnetwork_edges					<< ";"
			<< avg.num_secondary_roadnetwork_edges					<< ";"
			<< avg.num_collisions									<< ";"
			<< avg.memory_in_use									<< ";"
			<< std::endl;
	}

	out.close();
}

int main(int argc, char** argv)
{
	if (argc < 5)
	{
		std::cout << "command line options: <input file> <repetitions> <file=0/database=1> <output file>" << std::endl;
		exit(-1);
	}

	std::string inputFile = argv[1];
	unsigned int repetitions = atoi(argv[2]);
	unsigned int database = atoi(argv[3]) == 1;
	std::string outputFile = argv[4];

	try
	{
		std::string content = FileReader::read(inputFile);
		std::vector<std::string> records;
		StringUtils::tokenize(content, "\n", records);

		unsigned int numRecords = records.size() - 1;

		if (numRecords % repetitions != 0)
		{
			throw std::exception("repetitions param is not multiple of num. records in input file");
		}

		unsigned int numGroups = numRecords / repetitions;

		Group* groups = new Group[numGroups];
		for (unsigned int i = 0, j = 1; i < numGroups; i++)
		{
			Group& group = groups[i];
			group.allocate(repetitions);
			for (unsigned int k = 0; k < repetitions; k++)
			{
				std::string record = records[j++];
				std::vector<std::string> fields;
				StringUtils::tokenize(record, ";", fields);
				group.records[k].timestamp =							fields[ftimestamp											];
				group.records[k].config_name =							fields[fconfig_name											];
				group.records[k].expansion_kernel_blocks =				toInt(		fields[fexpansion_kernel_blocks					]);
				group.records[k].expansion_kernel_threads =				toInt(		fields[fexpansion_kernel_threads				]);
				group.records[k].collision_detection_kernel_blocks =	toInt(		fields[fcollision_detection_kernel_blocks		]);
				group.records[k].collision_detection_kernel_threads =	toInt(		fields[fcollision_detection_kernel_threads		]);
				group.records[k].max_highway_derivations =				toInt(		fields[fmax_highway_derivations					]);
				group.records[k].max_street_derivations =				toInt(		fields[fmax_street_derivations					]);
				group.records[k].quadtree_depth =						toInt(		fields[fquadtree_depth							]);
				group.records[k].primary_roadnetwork_expansion_time =	toFloat(	fields[fprimary_roadnetwork_expansion_time		]);	
				group.records[k].collisions_computation_time =			toFloat(	fields[fcollisions_computation_time				]);
				group.records[k].primitives_extraction_time = 			toFloat(	fields[fprimitives_extraction_time				]);
				group.records[k].secondary_roadnetwork_expansion_time =	toFloat(	fields[fsecondary_roadnetwork_expansion_time	]);
				group.records[k].memory_copy_gpu_cpu_time =				toFloat(	fields[fmemory_copy_gpu_cpu_time				]);
				group.records[k].memory_copy_cpu_gpu_time = 			toFloat(	fields[fmemory_copy_cpu_gpu_time				]);
				group.records[k].num_vertices = 						toInt(		fields[fnum_vertices							]);
				group.records[k].num_primary_roadnetwork_edges =		toInt(		fields[fnum_primary_roadnetwork_edges			]);
				group.records[k].num_secondary_roadnetwork_edges =		toInt(		fields[fnum_secondary_roadnetwork_edges			]);
				group.records[k].num_collisions = 						toLong(		fields[fnum_collisions							]);
				group.records[k].memory_in_use = 						toInt(		fields[fmemory_in_use							]);
			}
		}

		//////////////////////////////////////////////////////////////////////////

		if (database)
		{
			outputToDatabase(outputFile, numGroups, groups);
		}
		else
		{
			outputToFile(outputFile, numGroups, groups);
		}

		//////////////////////////////////////////////////////////////////////////

		for (unsigned int i = 0; i < numGroups; i++)
		{
			groups[i].deallocate();
		}

		std::cout << "success..." << std::endl;
	} 
	catch (std::exception& e)
	{
		std::cerr << "error: " << e.what() << std::endl;
	}

	system("pause");

	return -1;
}