#include <FileReader.h>
#include <StringUtils.h>

#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <exception>

#define MAX_REPETITIONS 1000

#define ftimestamp									0
#define fconfig_name								1
#define fexpansion_kernel_blocks					2
#define fexpansion_kernel_threads					3
#define fcollision_detection_kernel_blocks			4
#define fcollision_detection_kernel_threads			5
#define fprimary_roadnetwork_expansion_time			6
#define fcollisions_computation_time				7
#define fprimitives_extraction_time					8
#define fsecondary_roadnetwork_expansion_time		9
#define fmemory_copy_gpu_cpu_time					10
#define fmemory_copy_cpu_gpu_time					11
#define fnum_vertices								12
#define fnum_edges									13
#define fnum_collisions								14
#define fmemory_in_use								15

struct Record
{
	std::string		timestamp;
	std::string		config_name;
	unsigned int	expansion_kernel_blocks;
	unsigned int	expansion_kernel_threads;
	unsigned int	collision_detection_kernel_blocks;
	unsigned int	collision_detection_kernel_threads;
	float			primary_roadnetwork_expansion_time;
	float			collisions_computation_time;
	float			primitives_extraction_time;
	float			secondary_roadnetwork_expansion_time;
	float			memory_copy_gpu_cpu_time;
	float			memory_copy_cpu_gpu_time;
	unsigned int	num_vertices;
	unsigned int	num_edges;
	unsigned long	num_collisions;
	unsigned int	memory_in_use;

	Record() :
		timestamp(""),
		config_name(""),
		expansion_kernel_blocks(0),
		expansion_kernel_threads(0),
		collision_detection_kernel_blocks(0),
		collision_detection_kernel_threads(0),
		primary_roadnetwork_expansion_time(0),
		collisions_computation_time(0),
		primitives_extraction_time(0),
		secondary_roadnetwork_expansion_time(0),
		memory_copy_gpu_cpu_time(0),
		memory_copy_cpu_gpu_time(0),
		num_vertices(0),
		num_edges(0),
		num_collisions(0),
		memory_in_use(0)
	{
	}

};

struct Group
{
	unsigned int	repetitions;
	Record*			records;
	/*std::string		*timestamp;
	std::string		*config_name;
	unsigned int	*expansion_kernel_blocks;
	unsigned int	*expansion_kernel_threads;
	unsigned int	*collision_detection_kernel_blocks;
	unsigned int	*collision_detection_kernel_threads;
	float			*primary_roadnetwork_expansion_time;
	float			*collisions_computation_time;
	float			*primitives_extraction_time;
	float			*secondary_roadnetwork_expansion_time;
	float			*memory_copy_gpu_cpu_time;
	float			*memory_copy_cpu_gpu_time;
	unsigned int	*num_vertices;
	unsigned int	*num_edges;
	unsigned long	*num_collisions;
	unsigned int	*memory_in_use;*/

	void allocate(unsigned int repetitions)
	{
		this->repetitions						= repetitions;
		this->records							= new Record[repetitions];
		/*timestamp								= new std::string[repetitions];
		config_name								= new std::string[repetitions];
		expansion_kernel_blocks					= new unsigned int[repetitions];
		expansion_kernel_threads				= new unsigned int[repetitions];
		collision_detection_kernel_blocks		= new unsigned int[repetitions];
		collision_detection_kernel_threads		= new unsigned int[repetitions];
		primary_roadnetwork_expansion_time		= new float[repetitions];
		collisions_computation_time				= new float[repetitions];
		primitives_extraction_time				= new float[repetitions];
		secondary_roadnetwork_expansion_time	= new float[repetitions];
		memory_copy_gpu_cpu_time				= new float[repetitions];
		memory_copy_cpu_gpu_time				= new float[repetitions];
		num_vertices							= new unsigned int[repetitions];
		num_edges								= new unsigned int[repetitions];
		num_collisions							= new unsigned long[repetitions];
		memory_in_use							= new unsigned int[repetitions];*/
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
			avg.primary_roadnetwork_expansion_time		+= records[i].primary_roadnetwork_expansion_time	;
			avg.collisions_computation_time				+= records[i].collisions_computation_time			;
			avg.primitives_extraction_time				+= records[i].primitives_extraction_time			;
			avg.secondary_roadnetwork_expansion_time	+= records[i].secondary_roadnetwork_expansion_time	;
			avg.memory_copy_gpu_cpu_time				+= records[i].memory_copy_gpu_cpu_time				;
			avg.memory_copy_cpu_gpu_time				+= records[i].memory_copy_cpu_gpu_time				;
			avg.num_vertices							+= records[i].num_vertices							;
			avg.num_edges								+= records[i].num_edges								;
			avg.num_collisions							+= records[i].num_collisions						;
			avg.memory_in_use							+= records[i].memory_in_use							;
		}

		avg.expansion_kernel_blocks					/= repetitions;
		avg.expansion_kernel_threads				/= repetitions;
		avg.collision_detection_kernel_blocks		/= repetitions;
		avg.collision_detection_kernel_threads		/= repetitions;
		avg.primary_roadnetwork_expansion_time		/= repetitions;
		avg.collisions_computation_time				/= repetitions;
		avg.primitives_extraction_time				/= repetitions;
		avg.secondary_roadnetwork_expansion_time	/= repetitions;
		avg.memory_copy_gpu_cpu_time				/= repetitions;
		avg.memory_copy_cpu_gpu_time				/= repetitions;
		avg.num_vertices							/= repetitions;
		avg.num_edges								/= repetitions;
		avg.num_collisions							/= repetitions;
		avg.memory_in_use							/= repetitions;

		return avg;
	}

	void deallocate()
	{
		delete[] records;
		/*delete[] timestamp;
		delete[] config_name;
		delete[] expansion_kernel_blocks;
		delete[] expansion_kernel_threads;
		delete[] collision_detection_kernel_blocks;
		delete[] collision_detection_kernel_threads;
		delete[] primary_roadnetwork_expansion_time;
		delete[] collisions_computation_time;
		delete[] primitives_extraction_time;
		delete[] secondary_roadnetwork_expansion_time;
		delete[] memory_copy_gpu_cpu_time;
		delete[] memory_copy_cpu_gpu_time;
		delete[] num_vertices;
		delete[] num_edges;
		delete[] num_collisions;
		delete[] memory_in_use;*/
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

#define toInt(x) atoi(x.c_str())
#define toLong(x) atol(x.c_str())

int main(int argc, char** argv)
{
	if (argc < 4)
	{
		std::cout << "command line options: <input file> <repetitions> <output file>" << std::endl;
		exit(-1);
	}

	try
	{
		std::string content = FileReader::read(argv[1]);
		std::vector<std::string> records;
		StringUtils::tokenize(content, "\n", records);

		unsigned int repetitions = atoi(argv[2]);

		std::string outputFile = argv[3];

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
				group.records[k].timestamp =							fields[ftimestamp								];
				group.records[k].config_name =							fields[fconfig_name								];
				group.records[k].expansion_kernel_blocks =				toInt(		fields[fexpansion_kernel_blocks					]);
				group.records[k].expansion_kernel_threads =				toInt(		fields[fexpansion_kernel_threads				]);
				group.records[k].collision_detection_kernel_blocks =	toInt(		fields[fcollision_detection_kernel_blocks		]);
				group.records[k].collision_detection_kernel_threads =	toInt(		fields[fcollision_detection_kernel_threads		]);
				group.records[k].primary_roadnetwork_expansion_time =	toFloat(	fields[fprimary_roadnetwork_expansion_time		]);	
				group.records[k].collisions_computation_time =			toFloat(	fields[fcollisions_computation_time				]);
				group.records[k].primitives_extraction_time = 			toFloat(	fields[fprimitives_extraction_time				]);
				group.records[k].secondary_roadnetwork_expansion_time =	toFloat(	fields[fsecondary_roadnetwork_expansion_time	]);
				group.records[k].memory_copy_gpu_cpu_time =				toFloat(	fields[fmemory_copy_gpu_cpu_time				]);
				group.records[k].memory_copy_cpu_gpu_time = 			toFloat(	fields[fmemory_copy_cpu_gpu_time				]);
				group.records[k].num_vertices = 						toInt(		fields[fnum_vertices							]);
				group.records[k].num_edges = 							toInt(		fields[fnum_edges								]);
				group.records[k].num_collisions = 						toLong(		fields[fnum_collisions							]);
				group.records[k].memory_in_use = 						toInt(		fields[fmemory_in_use							]);
			}
		}

		//////////////////////////////////////////////////////////////////////////

		std::ofstream out;
		out.open(outputFile.c_str(), std::ios::out);

		out << "timestamp;" 
			<< "config_name;" 
			<< "expansion_kernel_blocks;" 
			<< "expansion_kernel_threads;" 
			<< "collision_detection_kernel_blocks;" 
			<< "collision_detection_kernel_threads;" 
			<< "primary_roadnetwork_expansion_time;" 
			<< "collisions_computation_time;" 
			<< "primitives_extraction_time;" 
			<< "secondary_roadnetwork_expansion_time;" 
			<< "memory_copy_gpu_cpu_time;" 
			<< "memory_copy_cpu_gpu_time;" 
			<< "num_vertices;" 
			<< "num_edges;" 
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
				<< toString(avg.primary_roadnetwork_expansion_time)		<< ";"
				<< toString(avg.collisions_computation_time)			<< ";"
				<< toString(avg.primitives_extraction_time)				<< ";"
				<< toString(avg.secondary_roadnetwork_expansion_time)	<< ";"
				<< toString(avg.memory_copy_gpu_cpu_time)				<< ";"
				<< toString(avg.memory_copy_cpu_gpu_time)				<< ";"
				<< avg.num_vertices										<< ";"
				<< avg.num_edges										<< ";"
				<< avg.num_collisions									<< ";"
				<< avg.memory_in_use									<< ";"
				<< std::endl;
		}

		out.close();

		//////////////////////////////////////////////////////////////////////////

		for (unsigned int i = 0; i < numGroups; i++)
		{
			groups[i].deallocate();
		}

		return 0;
	} 
	catch (std::exception& e)
	{
		std::cerr << "error: " << e.what() << std::endl;
		system("pause");
	}

	return -1;
}